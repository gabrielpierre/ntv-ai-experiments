"""Isolated frozen-model inference, compatible with the common Detection contract."""
import math
import time

from ntv_ai.experiments.workshop_prompts import CLASSES, grounding_text, qwen_text
from ntv_ai.experiments.workshop_metrics import parse_qwen
from ntv_ai.models.inference import detections


class VlmPredictor:
    def __init__(self, name, path, resolution, prompt_id):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        self.name, self.resolution, self.prompt_id = name, resolution, prompt_id
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.device = torch.device('cuda:0')
        if name == 'grounding_dino':
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True,
                                                                           dtype=torch.float32).to(self.device).eval()
        else:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, local_files_only=True,
                quantization_config=quant, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, image):
        import torch
        from PIL import Image
        size = image.size
        image = image.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result, raw = self._grounding(image, size) if self.name == 'grounding_dino' else self._qwen(image, size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        raw.update(seconds=elapsed, received_size=list(size), model_resolution=self.resolution)
        return result, elapsed, raw

    def _grounding(self, image, size):
        import torch
        text, spans = grounding_text(self.prompt_id)
        tokenized = self.processor.tokenizer(text, return_offsets_mapping=True)
        maximum = self.model.config.max_text_len
        if len(tokenized['input_ids']) > maximum:
            raise ValueError(f'Prompt truncation: {len(tokenized["input_ids"])} > {maximum}')
        inputs = self.processor(images=image, text=text, return_tensors='pt', do_resize=False).to(self.device)
        if tuple(inputs['pixel_values'].shape[-2:]) != (self.resolution, self.resolution):
            raise ValueError('Grounding DINO changed image resolution')
        if inputs['input_ids'][0].tolist() != tokenized['input_ids']:
            raise ValueError('Grounding tokenizer changed the frozen text')
        outputs = self.model(**inputs)
        probabilities = outputs.logits[0].sigmoid()
        class_scores = []
        for a, b in spans:
            indices = [i for i, (x, y) in enumerate(tokenized['offset_mapping'])
                       if y > x and x < b and y > a and any(c.isalnum() for c in text[x:y])]
            if not indices:
                raise ValueError('Class has no text tokens')
            class_scores.append(probabilities[:, indices].max(dim=1).values)
        score, label = torch.stack(class_scores, dim=1).max(dim=1)
        center, wh = outputs.pred_boxes[0][:, :2], outputs.pred_boxes[0][:, 2:]
        boxes = torch.cat([center - wh / 2, center + wh / 2], dim=1)
        boxes = boxes * torch.tensor([*size, *size], device=self.device)
        result = detections(boxes.tolist(), score.tolist(), label.tolist(), CLASSES, size, .001, 100, clip=True)
        return result, {'text': text, 'score_kind': 'native_max_token_similarity',
                        'queries': {'boxes_xyxy': boxes.tolist(), 'scores': score.tolist(), 'labels': label.tolist()},
                        'invalid_outputs': []}

    def _qwen(self, image, size):
        import torch
        text = qwen_text(self.prompt_id)
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': text}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt', min_pixels=self.resolution**2,
            max_pixels=self.resolution**2).to(self.device)
        grid = inputs['image_grid_thw'][0].tolist()
        patch = self.model.config.vision_config.patch_size
        if grid != [1, self.resolution//patch, self.resolution//patch]:
            raise ValueError(f'Qwen changed image resolution: grid={grid}, patch={patch}')
        outputs = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4096,
            return_dict_in_generate=True, output_scores=True, use_cache=True)
        ids = outputs.sequences[0, inputs['input_ids'].shape[-1]:].tolist()
        # Extract only probabilities of emitted tokens; never retain full-vocabulary scores on disk.
        logprobs = [float(torch.log_softmax(score[0].float(), dim=-1)[token]) for score, token in zip(outputs.scores, ids)]
        tokenizer = self.processor.tokenizer
        parts = [tokenizer.decode([token], skip_special_tokens=True, clean_up_tokenization_spaces=False) for token in ids]
        answer = ''.join(parts)
        decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if answer != decoded:
            raise ValueError('Token-to-character alignment is not lossless')
        spans, offset = [], 0
        for part in parts:
            spans.append((offset, offset + len(part)))
            offset += len(part)
        result, rejected = parse_qwen(answer, spans, logprobs, CLASSES, size)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        if len(ids) == 4096 and ids[-1] not in eos:
            result = []
            rejected.append({'reason': 'generation_truncated'})
        raw = {'text': text, 'response': answer, 'token_ids': ids, 'token_spans': spans,
               'token_logprobs': logprobs, 'invalid_outputs': rejected,
               'score_kind': 'geometric_mean_class_and_coordinate_token_probability_uncalibrated',
               'image_grid_thw': grid, 'new_tokens': len(ids)}
        del outputs
        return result, raw

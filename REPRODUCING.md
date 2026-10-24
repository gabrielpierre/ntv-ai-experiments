# Reproducing the paper

## 1. Data and experimental boundaries

The main benchmark uses 30 training images (593 boxes), 8 validation images (160 boxes), and 18 test images (411 boxes). Training and validation are unchanged from the 50-image pilot. Four pilot images are temporal gaps. The test is the original 8-image test (138 boxes) plus 10 manually selected, corrected images (273 boxes), outside the evaluated detectors' training and validation. Its capture counts are 13/2/3 for 001/002/003. The models were kept fixed for this expanded evaluation.

The external, combined-source, and VLM experiments retain the original 8-image test. Do not replace it with the 18-image test when reproducing their published numbers.

| External source | Train | Validation | Test | Detection classes |
| --- | ---: | ---: | ---: | --- |
| MPID | 65 | 18 | 17 | Insulator |
| PTL-AI Furnas | 65 | 18 | 17 | Insulator, spacer, damper, signage |
| STN PLAD | 63 | 17 | 17 | Tower, insulator, spacer, damper, signs |
| TTPLA | 58 | 15 | 15 | Lattice tower |

TTPLA's 12 `void` images are excluded; other original annotations remain recorded but do not enter the mapped task. The historical Furnas mapping groups marker balls with signs; these objects are not semantically equivalent. The combined-source study therefore uses only insulator, spacer, and damper. Its training arms contain 30, 128, and 158 images, and all select checkpoints on the same eight NTV-AI validation images.

Request the NTV-AI images from the maintainers. External image download definitions are in `configs/downloads.yaml`; the source repositories are [MPID](https://github.com/phd-benel/MPID), [STN PLAD](https://github.com/andreluizbvs/PLAD), [PTL-AI Furnas](https://github.com/freds0/PTL-AI_Furnas_Dataset), and [TTPLA](https://github.com/R3ab/ttpla_dataset).

```bash
python -m pip install -e '.[download,evaluation,dev]'
python scripts/download_datasets.py --datasets mpid stn_plad ptl_ai ttpla
```

Locate the exact frozen images by hash. Use a destination outside the searched folders; originals are never overwritten:

```bash
python scripts/stage_paper_images.py --study all \
  --search /path/to/ntv-images --search data/external \
  --output /path/to/paper-images
python scripts/paper.py prepare --study all \
  --roots /path/to/paper-images/roots.json --output prepared/paper
```

Missing images are reported by SHA-256; no replacement sample is silently selected. Alternatively, supply a JSON mapping from source identifiers (`ntv_ai`, `ntv_review_extension`, `ntv_anotador_v2`, `mpid`, `ptl_ai`, `stn_plad`, `ttpla`) to image roots. Relative image paths are listed in each snapshot's `splits.json`. Use `--study main`, `external`, `mixed`, `annotation`, or `vlm` to prepare only the required data.

Preparation preserves the frozen splits. It regenerates 1,280-pixel tiles with 25% overlap, edge alignment, clipped boxes, and empty tiles retained. Where historical tile hashes exist, regenerated images and box geometry are checked against them. **Use Pillow 12.3.0**, as in the original runs. The resulting `prepared/paper/catalog.json` contains relocated descriptors; no source annotation file is modified.

## 2. Linux environments and official pretrained weights

The experiments use separate Conda environments for the three detectors:

```bash
conda env create -f arquiteturas/yolo26/environment.yml
conda env create -f arquiteturas/dfine/environment.yml
conda env create -f arquiteturas/rfdetr/environment.yml
conda run -n ntv-yolo26l python -m pip install -e '.[evaluation]'
conda run -n ntv-dfine python -m pip install -e '.[evaluation]'
conda run -n ntv-rfdetr python -m pip install -e '.[evaluation]'
python -m ntv_ai.cli acquire --config configs/experiments/assets.yaml --storage-root .
```

`arquiteturas/*/requirements.lock.txt`, `environment.lock.yml`, and `conda-linux-64.lock.txt` record the original environments. Important versions are Python 3.12.14, PyTorch 2.10.0/CUDA 12.8, Ultralytics 8.4.155, D-FINE revision `956d1709314c2c6a4df6f34de232054578a7449f`, and RF-DETR 1.11.0. Asset acquisition uses pinned URLs, revisions, and SHA-256 hashes. It also acquires the small COCO8 software fixture; that fixture is not training data for the paper.

Official initialization files are `yolo26s.pt`, `dfine_s_coco.pth`, and `rf-detr-nano.pth`. They are distinct from the trained pilot package. Training starts from the official COCO weights in every seed/protocol combination.

## 3. Training and checkpoint selection

List exact identifiers:

```bash
python scripts/paper.py list
```

Example main-benchmark run:

```bash
python scripts/paper.py train \
  --id main/rfdetr-nano-tiles1280-seed43 \
  --prepared prepared/paper/catalog.json --runs runs/paper
```

Add `--dry-run` to inspect the resolved recipe without training. Repeat for the 18 `main/` identifiers, 72 `external/` identifiers, and 18 `mixed/` identifiers needed for the respective studies. The annotation detector has a separate `annotation/` identifier. `paper/experiments.json` contains the exact values; do not substitute the generic full-dataset campaign.

The common protocol uses batch size 1, FP16 training, EMA selection, no random image augmentations, and seeds 42/43/44. Warm-up is three epochs, followed by validation-AP plateau learning-rate reduction and early stopping after 25 epochs. The maximum is 10,000 epochs; all historical scientific runs stopped earlier. The annotation detector's maximum is 100 epochs. Full-image and tiled protocols have different numbers of updates per epoch.

To resume an interrupted run, provide `--resume runs/paper/ACTUAL_RUN_DIRECTORY` with the same experiment identifier and prepared data. The existing checkpoint/control state is reused. An already completed early-stopped run is not restarted by resetting patience.

Run directories retain resolved configuration, runtime information, optimizer/control state, validation metrics, selected checkpoint, and predictions. Numerical equivalence of a new GPU training run depends on the pinned software and hardware; the exact published predictions remain available for metric replay.

## 4. Evaluate newly trained checkpoints

Use the training run directory printed by the training command:

```bash
python scripts/paper.py evaluate \
  --id main/rfdetr-nano-tiles1280-seed43 \
  --source-run runs/paper/ACTUAL_RUN_DIRECTORY \
  --prepared prepared/paper/catalog.json --runs runs/evaluations
```

For `main/` identifiers, this evaluates the 18-image test. For `external/`, it evaluates the source test and original 8-image NTV-AI test. For `mixed/`, it evaluates the three original target tests. Targets and class intersections are frozen in the experiment definition. Training/validation overlap is checked before cross-dataset evaluation.

COCOeval uses IoUs 0.50–0.95, confidence 0.001, at most 100 detections per image, FP32 inference, and batch size 1. Tiled predictions are projected into original coordinates and consolidated with class-wise NMS at IoU 0.50. NMS does not merge fragmented structures. Metrics use original-image box areas.

## 5. Evaluate the published trained models directly

Download the [original pilot model package](https://drive.google.com/file/d/1_7_dMiDW62qesVG9k1e81qCB1DX4cR03/view?usp=sharing) and extract it into `data/modelos/modelos-ntv-ai-20260929`. It contains the 18 selected weights and full resumable checkpoints. Their published hashes are recorded in `paper/experiments.json`.

After preparing the data, export the expanded-test descriptor and use the existing package evaluator:

```bash
python scripts/paper.py descriptor --view expanded18 --protocol tiles1280 \
  --prepared prepared/paper/catalog.json --output prepared/expanded18-tiles1280.json
python -m ntv_ai.cli models evaluate \
  --package data/modelos/modelos-ntv-ai-20260929 \
  --model rfdetr-nano --protocol tiles1280 --seed 43 \
  --target-dataset prepared/expanded18-tiles1280.json \
  --runs-root runs/published-checkpoints
```

The package evaluator checks its checkpoint hashes. Large weights for external, mixed-source, and annotation runs are not embedded in Git; reproduce those runs from the included configurations or request the corresponding checkpoints from the maintainers.

## 6. Language-guided models

Use a separate environment with the recorded packages in `paper/vlm-requirements.freeze.txt`; install this project there as well. Use `--extra-index-url https://download.pytorch.org/whl/cu128` when installing that freeze file. Its local Conda build locator for `packaging` has been replaced with the verified installed version, 26.3. `paper/vlm-assets.json` records the exact Hugging Face revisions and per-file hashes for Grounding DINO Tiny and Qwen3-VL-4B-Instruct. Download those revisions with `huggingface_hub.snapshot_download`, preserving every listed file. The two models use FP32 and NF4/BF16, respectively; Qwen uses greedy generation, one beam, and at most 4,096 new tokens.

Run the frozen `vlm/...` identifiers printed by `list` in that environment:

```bash
python scripts/paper.py vlm --id vlm/qwen/tiles1280/val/0 \
  --prepared prepared/paper/catalog.json --asset /path/to/qwen-snapshot \
  --output runs/vlm/qwen-tiles-val-p0
```

Run validation configurations first. The frozen test configurations reproduce the original selected prompts and thresholds, alongside P0; they do not select prompts using test outcomes. Prompt texts are in `paper/prompts.json` and `src/ntv_ai/experiments/workshop_prompts.py`. Invalid model output remains in the evaluation, while infrastructure failures stop the run. Per-image caching supports resuming the same request after interruption.

## 7. Annotation development and dataset statistics

The `annotation/` recipe reproduces the 68/10 development run from the deduplicated 78-image reference set. This set includes images from the pilot test and must not be evaluated as an independent model on that old test. The later 81/20 annotator is outside this release.

The historical class-wise combination uses pilot RF-DETR-Nano `full640`, seed 43, for towers; pilot `tiles1280`, seed 43, for foundations and spacers; and the 68-image fine-tuned model for insulators, dampers, and signs. Its export confidence is 0.10; class outputs are concatenated without cross-class NMS or box averaging. Supplementary Table S9 and the frozen combination results document this development comparison.

`python scripts/paper.py replay-annotation` recomputes the five saved development comparisons, including Table S9, without images or a GPU. Proposal generation used different qualified inference settings: the historical tiled proposer uses FP32, batch 4, and matrix-multiplication TF32; the new proposer uses FP16; the full-image proposer uses FP32. These operational settings are recorded in `paper/results/annotation-combination.json` and must not be substituted for the scientific benchmark's FP32/batch-1 settings.

The released statistics describe 4,427 images and 110,917 boxes. The full expansion is not used as the main benchmark's training set. Its class/capture counts and relative-box-area percentiles are in `paper/results/estatisticas-dataset-20261023.json` and `tamanhos-percentuais-20261023.json`.

## 8. Software verification

```bash
python -m pip install -e '.[evaluation,dev,campaign]'
python -m pytest -m 'not gpu'
python scripts/paper.py verify
```

GPU integration tests require explicit opt-in and are not run by metric replay. No training, dataset download, or bulk annotation starts when this package is installed or imported.

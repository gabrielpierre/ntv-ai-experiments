"""Frozen descriptions: all families preserve the seven existing annotation classes."""
CLASSES = ['torre', 'isolador', 'fundacao_concreto', 'fundacao_estai',
           'espacador', 'amortecedor', 'sinalizacao']
NAMES = ['transmission tower', 'insulator', 'concrete foundation', 'guy wire foundation',
         'conductor spacer', 'vibration damper', 'sign plate']
TECHNICAL = ['lattice transmission tower', 'power line insulator string', 'tower concrete footing',
             'guy wire ground anchor foundation', 'bundled conductor spacer',
             'Stockbridge vibration damper', 'tower warning or identification plate']
VISUAL = ['metal framework with crossed beams', 'a chain of repeated insulating discs or ribs',
          'concrete base supporting a tower leg', 'ground foundation where a diagonal guy wire ends',
          'a rigid fitting connecting parallel conductors', 'paired metal weights linked by a short cable',
          'a flat sign with printed symbols or numbers']
CONTEXT = ['carrying overhead power lines', 'between an energized conductor and its support',
           'at ground level beneath the tower', 'offset from the tower and attached to a guy wire',
           'holding bundled conductors apart', 'clamped onto an overhead conductor',
           'attached to the tower structure']


def phrases(prompt_id):
    if prompt_id not in range(4):
        raise ValueError('Unknown frozen prompt')
    if prompt_id == 0:
        return NAMES.copy()
    if prompt_id == 1:
        return TECHNICAL.copy()
    if prompt_id == 2:
        return [f'{a}, {b}' for a, b in zip(TECHNICAL, VISUAL)]
    return [f'{a}, {b}, {c}' for a, b, c in zip(TECHNICAL, VISUAL, CONTEXT)]


def grounding_text(prompt_id):
    parts, spans, position = phrases(prompt_id), [], 0
    for part in parts:
        spans.append((position, position + len(part)))
        position += len(part) + 2
    return '. '.join(parts) + '.', spans


def qwen_text(prompt_id):
    definitions = '\n'.join(f'{c}: {p}' for c, p in zip(CLASSES, phrases(prompt_id)))
    return ('Locate every visible instance of these classes in this image.\n' + definitions +
            '\nReturn only a JSON object with key "detections", an array of objects with exactly '
            '"class_id" and "bbox_xyxy". Use only the class_id values listed above. '
            'Each bbox_xyxy is [x_min,y_min,x_max,y_max] in coordinates normalized to 0..1000 '
            'relative to this image. Enclose each visible object tightly. Do not infer hidden objects. '
            'Return {"detections":[]} if none is visible. Return at most 100 objects. '
            'Do not include scores, explanations, Markdown or additional keys.')


def bank():
    return {'language': 'English', 'classes': CLASSES, 'annotation_semantics': 'unchanged',
            'families': {str(i): {'phrases': phrases(i), 'grounding_text': grounding_text(i)[0],
                                  'qwen_text': qwen_text(i)} for i in range(4)}}

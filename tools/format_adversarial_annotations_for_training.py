import json

output_fn = 'data/adversarial_annotations/adversarial_annotations_for_training.jsonl'
answer_fn = 'data/modified_annots/train_answer_shortcut/train_answer_shortcut_v2.jsonl'


def _read_jsonl_file(input_fn):
  annots = []
  with open(input_fn, 'r') as f:
    for line in f:
      annots.append(json.loads(line))
  return annots


answer_annots = _read_jsonl_file(answer_fn)

with open(output_fn, 'w') as f:
  for answer_annot in answer_annots:
    annots = {
        'annot_id': answer_annot['annot_id'],
        'answer_label': answer_annot['label'],
        'answer_tokens': answer_annot['result_tokens'],
        'answer_losses': answer_annot['result_losses'],
    }
    f.write('%s\n' % json.dumps(annots))

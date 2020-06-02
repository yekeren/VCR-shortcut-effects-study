import json

output_fn = 'data/adversarial_annotations/adversarial_annotations.jsonl'
answer_fn = 'data/modified_annots/val_answer_shortcut/val_answer_shortcut.jsonl.new'
rationale_fn = 'data/modified_annots_rationale/val_answer_shortcut/val_answer_shortcut_rationale.jsonl.new'


def _read_jsonl_file(input_fn):
  annots = []
  with open(input_fn, 'r') as f:
    for line in f:
      annots.append(json.loads(line))
  return annots


answer_annots = _read_jsonl_file(answer_fn)
rationale_annots = _read_jsonl_file(rationale_fn)

with open(output_fn, 'w') as f:
  for answer_annot, rationale_annot in zip(answer_annots, rationale_annots):
    assert answer_annot['annot_id'] == rationale_annot['annot_id']
    annots = {
        'annot_id': answer_annot['annot_id'],
        'answer_label': answer_annot['label'],
        'answer_tokens': answer_annot['result_tokens'],
        'answer_losses': answer_annot['result_losses'],
        'rationale_label': rationale_annot['label'],
        'rationale_tokens': rationale_annot['result_tokens'],
        'rationale_losses': rationale_annot['result_losses'],
    }
    f.write('%s\n' % json.dumps(annots))

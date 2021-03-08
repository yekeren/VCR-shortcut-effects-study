# VCR-shortcut-effects-study

  * [Introduction](introduction)
  * [Validation data for verifying the shortcuts](#validation-data-for-verifying-the-shortcuts)
    -  [Rule-based modification](#rule-based-modification)
    -  [Adversarial modification](#adversarial-modification)
  * [Our paper](#our-paper)

## Introduction
Code and data of our AAAI2021 paper "A Case Study of the Shortcut Effects in Visual Commonsense Reasoning"

## Validation data for verifying the shortcuts

We provide the validation data to verify the shortcut effects.
Please refer to the links below to download the data.
The methodology to generate these data are mentioned in our paper section "Methods to Evaluate the Shortcut Effects".
We also provide more details below.

|                       | Setting (link)                                             | Count  | Used in                 |
|-----------------------|------------------------------------------------------------|--------|-------------------------|
| Rule-based Modified   | [Rule-Singular](data/rule_based/val_rule_singular.jsonl)   | 16,154 | Paper Table 3, Row 2    |
|                       | [Rule-Plural](data/rule_based/val_rule_plural.jsonl)       | 3,657  | Paper Table 3, Row 3    |
| Adversarially Modifed | [AdvTop-1](data/adversarial_based/val_adv_rmtop1.jsonl)    | 26,534 | Paper Table 4, Column 4 |
|                       | [KeepTop-1](data/adversarial_based/val_adv_keeptop1.jsonl) | 26,534 | Paper Table 4, Column 5 |
|                       | [KeepTop-3](data/adversarial_based/val_adv_keeptop3.jsonl) | 26,534 | Paper Table 4, Column 6 |
|                       | [KeepTop-5](data/adversarial_based/val_adv_keeptop5.jsonl) | 26,534 | Paper Table 4, Column 7 |

### Rule-based modification

Please refer to the code under the "tools" directory. 
[rephrase_choice_singular.py](tools/rephrase_choice_singular.py) and [rephrase_choice_plural.py](tools/rephrase_choice_plural.py) generate the Rule-Singular and Rule-Plural validation data, respectively. To run them, just type ```python rephrase_choice_singular.py``` and ```python rephrase_choice_plural.py``` with default arguments.

### Adversarial modification

We use [shortcut_main.py](modeling/shortcut_main.py) to score the effect of removing individual tokens in the answer/rationale.
Then, we use [format_adversarial_annotations.py](tools/format_adversarial_annotations.py) to merge the results from both answering model and rationale model.
Finally, we use [merge_adversarial_annotations.py](tools/merge_adversarial_annotations.py) to generate the setting of AdvTop-1, KeepTop-1, KeepTop-3, KeepTop-5, which are used in our Table 4.

Here is an example pipeline to generate [AdvTop-1](data/adversarial_based/val_adv_rmtop1.jsonl) setting.
We assume the original VCR validation data is located at ```data/vcr1annots/val.jsonl``` and the scoring of the shortcut effects is located at [data/adversarial_based/shortcut_scores.jsonl](data/adversarial_based/shortcut_scores.jsonl). The following command shall generate the ```data/adversarial_based/val_adv_rmtop1.jsonl.v2``` file required for the AdvTop-1 setting.

```
python "tools/merge_adversarial_annotations.py" \
  --logtostderr \
  --annotations_jsonl_file "data/vcr1annots/val.jsonl" \
  --adversarial_annotations_jsonl_file "data/adversarial_based/shortcut_scores.jsonl" \
  --output_jsonl_file "data/adversarial_based/val_adv_rmtop1.jsonl.v2" \
  --name "remove_shortcut"
```

## Our paper
If you found this repository useful or used our data for evaluation, please cite our paper

```
@InProceedings{Ye_2021_AAAI,
  author = {Ye, Keren and Kovashka, Adriana},
  title = {A Case Study of the Shortcut Effects in Visual Commonsense Reasoning},
  booktitle = {Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI)},
  month = {Feb},
  year = {2021}
}
```

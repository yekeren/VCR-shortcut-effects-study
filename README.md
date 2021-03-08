# VCR-shortcut-effects-study
Code and data of our AAAI2021 paper "A Case Study of the Shortcut Effects in Visual Commonsense Reasoning"

## Validation data for verifying the shortcuts

We provide the validation data to verify the shortcut effects.
Please refer to the links below to download the data.
The methodology to generate these data are mentioned in our paper section "Methods to Evaluate the Shortcut Effects"

|                       | Setting (link)                                             | Count  | Used in                 |
|-----------------------|------------------------------------------------------------|--------|-------------------------|
| Rule-based Modified   | [Rule-Singular](data/rule_based/val_rule_singular.jsonl)   | 16,154 | Paper Table 3, Row 2    |
|                       | [Rule-Plural](data/rule_based/val_rule_plural.jsonl)       | 3,657  | Paper Table 3, Row 3    |
| Adversarially Modifed | [AdvTop-1](data/adversarial_based/val_adv_rmtop1.jsonl)    | 26,534 | Paper Table 4, Column 4 |
|                       | [KeepTop-1](data/adversarial_based/val_adv_keeptop1.jsonl) | 26,534 | Paper Table 4, Column 5 |
|                       | [KeepTop-3](data/adversarial_based/val_adv_keeptop3.jsonl) | 26,534 | Paper Table 4, Column 6 |
|                       | [KeepTop-5](data/adversarial_based/val_adv_keeptop5.jsonl) | 26,534 | Paper Table 4, Column 7 |

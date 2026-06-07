# BabyLM Baseline Pre-training (Extended Fork)

This is a fork of [babylm/baseline-pretraining](https://github.com/babylm/baseline-pretraining), the official baseline training infrastructure for the [BabyLM shared task](https://babylm.github.io/). It is used as the pre-training and model extraction infrastructure for the weight distribution and initialization experiments in:

> **Searching for Nooks and Crannies: Geometric and Mechanistic Perspectives on Transformer Language Model Interpretability**
> Anna C. Marbut. PhD dissertation, University of Montana (2026).

For the original BabyLM setup instructions (environment, data layout, base training commands), see the [upstream README](https://github.com/babylm/baseline-pretraining).

## What This Fork Adds

### Non-standard model variants
The dissertation's initialization experiments compare standard RoBERTa pre-training against several non-standard pre-training procedures from the literature, to test whether geometric properties of the latent space predict performance independently of linguistic training task:

- **Alajrami random model** — randomly initialized weights, no pre-training ([Alajrami & Navigli, 2022](https://aclanthology.org/2022.acl-short.61/))
- **ASCII prediction model** — pre-trained to predict ASCII character values rather than masked tokens (non-linguistic pre-training objective)
- **Shuffled dataset variants** — pre-training on word- and sentence-shuffled corpora, disrupting sequential linguistic structure while preserving vocabulary distribution

These are defined in `src/babylm_baseline_train/models/alajrami_models.py` and registered in `src/babylm_baseline_train/models/helper.py`.

### Expanded training scale
- **RoBERTa-100M** — RoBERTa trained on the full 100M token BabyLM dataset (in addition to the baseline 10M)
- **1B dataset option** — support for training on the 1B token dataset
- **Random initialization baseline** — untrained model at initialization, used as the reference point for weight movement analysis

### Full checkpoint saving
The original baseline saves only the final model. This fork adds support for saving checkpoints at each epoch, enabling the weight movement analysis in [`masking_and_analysis`](https://github.com/amarbut/masking_and_analysis) — which tracks how individual weights move from their initialized values across training.

### HuggingFace conversion utility
`src/babylm_baseline_train/models/model_load_save.py` adds utilities to load a checkpoint from a specific epoch and convert it to HuggingFace `save_pretrained` format, making trained models compatible with the HuggingFace ecosystem for downstream evaluation and geometric analysis.

## Repository Structure (additions highlighted)

```
src/babylm_baseline_train/
  configs/
    BabyLM/
      exp_strict_mask.py       Modified to support additional model variants
      general.py               General training configuration
  datasets/
    babyLM.py                  Extended with shuffled dataset variants and 1B option
    babyLM_for_hf.py           HuggingFace-compatible dataset loader
  models/
    alajrami_models.py     [+] Non-standard model implementations (ASCII, random)
    helper.py              [+] Model factory functions for all variants
    model_load_save.py     [+] Epoch checkpoint loading + HuggingFace conversion
  train/
    tk_funcs.py            [+] Tokenizer utilities including pretrained tokenizer support

scripts/
  general_train.py         [+] Extended training script with full checkpoint saving
```

## Training the Extended Models

Training follows the same pattern as the upstream baseline. From the `scripts/` folder:

```bash
# Standard RoBERTa on 10M tokens
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 \
    general_train.py --setting "BabyLM/exp_strict_mask.py:roberta_s1"

# RoBERTa on 100M tokens
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 \
    general_train.py --setting "BabyLM/exp_strict_mask.py:roberta_100M"
```

## Converting Checkpoints to HuggingFace Format

After training, convert epoch checkpoints for use with HuggingFace:

```bash
python -m babylm_baseline_train.models.model_load_save \
    --model_loc <path_to_model_dir> \
    --epoch 20
```

This saves the model in HuggingFace format to `<model_loc>/hf_20/`, ready for geometric analysis with the tools in `masking_and_analysis`.

## Environment

Python 3.9, HuggingFace `transformers` and `datasets`. Install with:

```bash
pip install -e .
```

Also requires: [pt_framework](https://github.com/chengxuz/pt_framework)

Set `BABYLM_ROOT_DIR` to the directory where models and data will live. Data should be at `${BABYLM_ROOT_DIR}/datasets/` with subfolders `babylm_10M`, `babylm_100M`, `babylm_dev`, `babylm_test`.

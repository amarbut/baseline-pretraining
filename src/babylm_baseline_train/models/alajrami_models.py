"""
reference: https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/model.py
"""

import torch
import random

from transformers import (PreTrainedTokenizerBase, BatchEncoding)

# for debugging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class DataCollatorForAsciiValuePrediction:
    """Data collator used for ascii value prediction."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_prob: float):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        
        # create label reference
        vocab_dict = tokenizer.get_vocab()
        self.id_to_label = {}
        for token, token_id in vocab_dict.items():
            ascii_value = 0
            for char in token:
                ascii_value += ord(char)

            # ABC → [(ascii(A) + ascii(B) + ascii(C))] % 5
            self.id_to_label[token_id] = ascii_value % 5
        

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # mask tokens and create word masks (labels)
        masked_input_ids, masked_word_labels = self.mask_tokens(batch["input_ids"])

        return {"input_ids": masked_input_ids, "attention_mask": batch["attention_mask"],
                "labels": masked_word_labels}


    def mask_tokens(self, input_ids):
        """Prepare masked tokens and their labels."""
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # init
        masked_word_labels = torch.zeros_like(input_ids)
        input_id_sets = set(input_ids.view(-1).tolist())

        # create labels
        for input_id in input_id_sets:
            masked_word_labels[input_ids == input_id] = self.id_to_label[input_id]
        
        # create a mask
        probability_matrix = torch.full(masked_word_labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_word_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # replace tokens with [MASK]
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return input_ids, masked_word_labels



class DataCollatorForRandomValuePrediction:
    """Data collator used for random value prediction."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_prob: float):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        
        # create label reference
        vocab_dict = tokenizer.get_vocab()
        self.id_to_label = {}
        for token, token_id in vocab_dict.items():
            # random number between 0 and 4
            random_value = random.randint(0, 4)
            self.id_to_label[token_id] = random_value
        

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # mask tokens and create word masks (labels)
        masked_input_ids, masked_word_labels = self.mask_tokens(batch["input_ids"])

        return {"input_ids": masked_input_ids, "attention_mask": batch["attention_mask"],
                "labels": masked_word_labels}


    def mask_tokens(self, input_ids):
        """Prepare masked tokens and their labels."""
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # init
        masked_word_labels = torch.zeros_like(input_ids)
        input_id_sets = set(input_ids.view(-1).tolist())

        # create labels
        for input_id in input_id_sets:
            masked_word_labels[input_ids == input_id] = self.id_to_label[input_id]
        
        # create a mask
        probability_matrix = torch.full(masked_word_labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_word_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # replace tokens with [MASK]
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return input_ids, masked_word_labels
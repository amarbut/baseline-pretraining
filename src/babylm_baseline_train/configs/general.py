import babylm_baseline_train.models.helper as helper
from babylm_baseline_train.train.tk_funcs import\
        get_tokenizer_func
import functools
from transformers import DataCollatorForLanguageModeling
from itertools import product
import copy
from babylm_baseline_train.models.alajrami_models import (DataCollatorForAsciiValuePrediction,
                             DataCollatorForRandomValuePrediction)

models = ['roberta-large',
          'shuffle-sentence',
          'shuffle-corpus',
          'ascii',
          'rand']
pretrained = ['babylm-base',
              'babylm-test',
              'normal_init/hf_20']

def add_collate_fn_for_MLM(key_params, tokenizer):
    if 'add_train_loader_kwargs' not in key_params:
        key_params['add_train_loader_kwargs'] = {}
    key_params['add_train_loader_kwargs'].update(
            {'collate_fn': DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
                )})
    return key_params

def add_collate_fn_for_ascii(key_params, tokenizer):
    if 'add_train_loader_kwargs' not in key_params:
        key_params['add_train_loader_kwargs'] = {}
    key_params['add_train_loader_kwargs'].update(
            {'collate_fn': DataCollatorForAsciiValuePrediction(
                tokenizer = tokenizer,
                mask_prob = 0.15)})
    return key_params

def add_collate_fn_for_rand(key_params, tokenizer):
    if 'add_train_loader_kwargs' not in key_params:
        key_params['add_train_loader_kwargs'] = {}
    key_params['add_train_loader_kwargs'].update(
            {'collate_fn': DataCollatorForRandomValuePrediction(
                tokenizer=tokenizer,
                mask_prob=0.15,
                )})
    return key_params

def add_func_in_general(
        func_name,
        data_func,
        exp_name=None,
        seed=None,
        model_name=None,
        all_things=None,
        post_func=None,
        **kwargs):

    if exp_name is None:
        exp_name = func_name
    def _func(key_params):
        key_params = data_func(key_params)
        if model_name == '350m':
            key_params['get_model_func'] = functools.partial(
                    helper.get_opt_func, 
                    opt_model_size='350m')
        elif model_name == 'roberta-base':
            key_params['get_model_func'] = helper.get_roberta_func
        elif model_name in models:
            key_params['get_model_func'] = functools.partial(
                    helper.get_roberta_func,
                    model_name=model_name)
        elif model_name in pretrained:
            key_params['get_model_func'] = functools.partial(
                    helper.get_pretrained_func,
                    model_name=model_name)
        elif model_name is not None:
            raise NotImplementedError(f"model name: {model_name}")
        key_params['exp_id'] = exp_name
        key_params['seed'] = seed
        key_params.update(kwargs)
        if post_func is not None:
            key_params = post_func(key_params)
        return key_params

    if all_things is None:
        all_things = globals()
    all_things[func_name] = _func

add_func_in_general_for_opt = add_func_in_general

def get_general_data_func(
        data_func, tokenizer=None, 
        max_epochs=100, ckpt_save_interval=50,
        col_name=None):
    def _func(key_params):
        if col_name is not None:
            key_params['col_name'] = col_name
        if tokenizer is None:
            _tokenizer = get_tokenizer_func()
        else:
            _tokenizer = tokenizer
        key_params['get_dataset_func'] = functools.partial(
                    data_func,
                    tokenizer=_tokenizer)
        key_params['max_epochs'] = max_epochs
        key_params['ckpt_save_interval'] = ckpt_save_interval
        return key_params
    return _func

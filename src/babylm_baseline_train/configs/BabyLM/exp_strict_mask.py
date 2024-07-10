import babylm_baseline_train.datasets.babyLM as babyLM
from babylm_baseline_train.configs.general import\
        add_func_in_general, get_general_data_func,\
        add_collate_fn_for_MLM, add_collate_fn_for_ascii, add_collate_fn_for_rand
import functools
from itertools import product
import babylm_baseline_train.train.tk_funcs as tk_funcs


KWARGS = dict(
        all_things=globals(),
        specify_iter=[],
        )
DATA_KWARGS = dict(
        max_epochs=20, ckpt_save_interval=15,
        col_name='babyLM_10M')

pretrain_epochs = [5, 10, 20]
retrain_epochs = list(range(1,21))

def add_exp_seeds(
        exp_names, seeds, data_func,
        model_name='roberta-base',
        tokenizer=None,
        collator= add_collate_fn_for_MLM,
        specify_epoch=pretrain_epochs
        ):
    for exp_name, seed in zip(exp_names, seeds):
        if tokenizer is None:
            MLM_tokenizer = tk_funcs.get_roberta_tokenizer_func(
                    model_name=model_name)
        else:
            MLM_tokenizer = tokenizer
        add_func_in_general(
                func_name=exp_name,
                data_func=get_general_data_func(
                    data_func,
                    tokenizer=MLM_tokenizer,
                    **DATA_KWARGS),
                seed=seed,
                model_name=model_name,
                post_func=functools.partial(
                    collator,
                    tokenizer=MLM_tokenizer),
                specify_epoch=specify_epoch,
                **KWARGS)

add_exp_seeds(
        exp_names=[
            'roberta_s1',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M)

add_exp_seeds(
        exp_names=[
            'roberta_large_s1',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='roberta-large')


add_exp_seeds(
        exp_names=[
            'babylm-base',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='babylm-base')


add_exp_seeds(
        exp_names=[
            'babylm-test',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='babylm-test')

add_exp_seeds(
        exp_names=[
            'shuffle-sentence',
            ], 
        seeds=[1], 
        data_func=babyLM.get_sentence_shuffle)

add_exp_seeds(
        exp_names=[
            'shuffle-corpus',
            ], 
        seeds=[1], 
        data_func=babyLM.get_corpus_shuffle)

add_exp_seeds(
        exp_names=[
            'ascii',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        collator=add_collate_fn_for_ascii)

add_exp_seeds(
        exp_names=[
            'rand',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        collator=add_collate_fn_for_rand)

add_exp_seeds(
        exp_names=[
            'roberta-baby',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='roberta-base',
        specify_epoch=retrain_epochs)

add_exp_seeds(
        exp_names=[
            'normal_init',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='normal_init/hf_20',
        specify_epoch=retrain_epochs)


add_exp_seeds(
        exp_names=[
            'rt_shuffle-sentence',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='shuffle-sentence/hf_20',
        specify_epoch=retrain_epochs)


add_exp_seeds(
        exp_names=[
            'rt_shuffle-corpus',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='shuffle-corpus/hf_20',
        specify_epoch=retrain_epochs)


add_exp_seeds(
        exp_names=[
            'rt_rand',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='rand/hf_20',
        specify_epoch=retrain_epochs)


add_exp_seeds(
        exp_names=[
            'rt_ascii',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='ascii/hf_20',
        specify_epoch=retrain_epochs)


add_exp_seeds(
        exp_names=[
            'rt_shuffle-index',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='shuffle_index/hf_20',
        specify_epoch=retrain_epochs)
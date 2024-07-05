#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:39:57 2024

@author: anna
"""

import os
import ipdb
import setuptools
import torch
import transformers
import argparse

import babylm_baseline_train.train.tk_funcs as tk_funcs
import babylm_baseline_train.models.helper as helper


def load_roberta(model_loc, epoch):
    tokenizer = tk_funcs.get_roberta_tokenizer_func()
    model = helper.get_roberta_func(tokenizer=tokenizer)
    saved_model = torch.load(f'{model_loc}/epoch_{epoch}.pth', map_location=torch.device('cpu'))
    saved_model['state_dict'].pop('roberta.embeddings.token_type_ids')
    saved_model['state_dict'].pop('roberta.embeddings.position_ids')
    model.load_state_dict(saved_model['state_dict'])
    
    return model, tokenizer

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_loc', help = 'location of trained model', required = True)
    parser.add_argument('--epoch', help = 'epoch checkpoint to load', required = True)
    kwargs = vars(parser.parse_args())
    
    model, tokenizer = load_roberta(**kwargs)
    model.save_pretrained(f'{kwargs.model_loc}/hf_{kwargs.epoch}')
    tokenizer.save_pretrained(f'{kwargs.model_loc}/hf_{kwargs.epoch}')
    
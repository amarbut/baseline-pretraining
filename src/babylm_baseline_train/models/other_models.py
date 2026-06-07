#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:42:29 2024

@author: anna
"""

from transformers import RobertaForMaskedLM, RobertaTokenizer
import torch
import numpy as np
import random

#shuffle input embedding indices
m = RobertaForMaskedLM.from_pretrained('roberta_s1/hf_20')
shuff_emb = np.copy(m.get_input_embeddings().weight.data.numpy())
random.shuffle(shuff_emb)
m.set_input_embeddings(torch.nn.Embedding.from_pretrained(torch.FloatTensor(shuff_emb)))

t = RobertaTokenizer.from_pretrained('roberta-base')

m.save_pretrained("shuffle_index/hf_20")
t.save_pretrained("shuffle_index/hf_20")


#initialize with N(0,1)
m = RobertaForMaskedLM.from_pretrained('roberta-base')
for name, param in m.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        param.requires_grad = False
        noise = torch.randn(param.shape) #N(0,1) noise
        param *= 0
        param += noise
        param.requires_grad = True
        
t = RobertaTokenizer.from_pretrained('roberta-base')

m.save_pretrained("/media/anna/Samsung_T5/Initialization/BabyLM/models/stand_norm/hf_20")
t.save_pretrained("media/anna/Samsung_T5/Initialization/BabyLM/models/stand_norm/hf_20")

m3 = RobertaForMaskedLM.from_pretrained("roberta-base")
alpha = 0.5
#alpha = 0 is no noise, 100 is all noise 
for name, param in m3.named_parameters():
    if name[8:15] == "encoder" and name[-6:] == "weight":
        param.requires_grad = False
        noise = (alpha/100)*torch.randn(param.shape) #N(0,1) noise
        param *= (1-(alpha/100))
        param += noise
        param.requires_grad = True
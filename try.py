from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig
import torch
import torch.nn as nn
from models.PrefixGPT2 import PrefixGPT2
from models.training_utils import get_conditional_prob_dist_fast, get_conditional_prob_dist_slow

config = AutoConfig.from_pretrained('gpt2')
config.pre_seq_len = 10
config.prefix_hidden_size = 800
config.prefix_projection = True
config.hidden_dropout_prob = 0.1
model = PrefixGPT2(config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

prompts = ['The doctor performing surgery is a young', 'The criminal is a black']
words_list = ['man', 'woman']
word_list = ['woman', 'man']
con_p_d = get_conditional_prob_dist_slow(model, tokenizer, prompts, words_list)
print(con_p_d)


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained('gpt2')
config.pre_seq_len = 10
config.prefix_hidden_size = 800
config.prefix_projection = True
config.hidden_dropout_prob = 0.1
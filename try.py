from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig
import torch
import torch.nn as nn
from models.PrefixGPT2 import PrefixGPT2
from models.training_utils import get_conditional_prob_dist
from models.training_utils import get_sent_prob_diff_loss, KLD


model = PrefixGPT2.from_pretrained("/home/CE/yifwang/master_thesis/debias_NLG/fast_model_use_full_words_True_alpha1_25_alpha2_50_alpha3_25_alpha4_100_beta_10_pre_seq_len_10_hidden_size_800_batch_size_16_lm_batch_size_128_lr_5e-05_lm_lr_5e-05_epochs_2_lm_epochs_0_warmup_500_lm_warmup_100_random_seed_42/epoch_2")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "The doctor is a young"
input = tokenizer(input_text, return_tensors='pt')
past_key_values =  model(**input, use_cache=True).past_key_values

# w/o past_k_v
input_text = "The doctor is a young black"
input = tokenizer(input_text, return_tensors='pt')
output = model(**input)
print(output.logits[-1])
# w past_k_v
input_text = "black"
input = tokenizer(input_text, return_tensors='pt')
output =  model(**input, past_key_values=past_key_values)
print(output.logits[-1])
# w/o past_k_v
input = tokenizer(input_text, return_tensors='pt')
output =  model(**input)
print(output.logits[-1])
# w past_k_v and mask=0
mask_length = past_key_values[0][0].shape[-2]
output =  model(**input)
print(output.logits[-1])
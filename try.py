from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig
import torch
import torch.nn as nn
from models.PrefixGPT2 import PrefixGPT2
from models.training_utils import get_conditional_prob_dist
from models.training_utils import get_sent_prob_diff_loss, KLD
config = AutoConfig.from_pretrained('gpt2')
config.pre_seq_len = 10
config.prefix_hidden_size = 800
config.prefix_projection = True
config.hidden_dropout_prob = 0.1
model = PrefixGPT2(config).cuda()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
kld_model = KLD()

prompts = ['The doctor performing surgery is a young','The doctor performing surgery is a young', 'The criminal is a black', 'i hate dog']
# print(get_sent_prob_diff_loss(model, tokenizer, prompts, kld_model))
word_list = ['man', 'woman']
prob_dist = get_conditional_prob_dist(model, tokenizer, prompts, word_list, 4)
print(prob_dist / torch.sum(prob_dist,dim=1).unsqueeze(1).expand(-1, prob_dist.shape[1]))
prob_dist = get_conditional_prob_dist(model, tokenizer, prompts, word_list, 3)
print(prob_dist / torch.sum(prob_dist,dim=1).unsqueeze(1).expand(-1, prob_dist.shape[1]))
prob_dist = get_conditional_prob_dist(model, tokenizer, prompts, word_list, 2)
print(prob_dist / torch.sum(prob_dist,dim=1).unsqueeze(1).expand(-1, prob_dist.shape[1]))
prob_dist = get_conditional_prob_dist(model, tokenizer, prompts, word_list, 1)
print(prob_dist / torch.sum(prob_dist,dim=1).unsqueeze(1).expand(-1, prob_dist.shape[1]))

from models.PrefixGPT2 import PrefixGPT2
from transformers import AutoConfig, GPT2Tokenizer
import torch
# initialize model config
config = AutoConfig.from_pretrained('gpt2')
config.pre_seq_len = 10
config.prefix_hidden_size = 800
config.prefix_projection = True
config.hidden_dropout_prob = 0.1

# initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# initialize prefix model
# model = PrefixGPT2(config)
# model.save_pretrained("model1.pt", model.prefix_encoder.state_dict())
model= PrefixGPT2.from_pretrained('model1.pt')
print(model)
#model = model.cuda()
sent = ['This is a young man.', 'The older doctor there is a black woman.']
inputs = tokenizer(sent, padding=True, return_tensors='pt')
#inputs = inputs.cuda()
labels = inputs['input_ids'].clone()
labels.masked_fill_(inputs['attention_mask'] == 0, -100)

pkv = model(**inputs, labels=labels).past_key_values
inputs["attention_mask"] = torch.cat((inputs["attention_mask"],inputs["attention_mask"]),dim=1)
print(model(**inputs, labels=labels, past_key_values=pkv).past_key_values[0][0].shape)

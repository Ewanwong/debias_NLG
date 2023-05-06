from model.PrefixGPT2 import PrefixGPT2
from transformers import AutoConfig, GPT2Tokenizer

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
model = PrefixGPT2(config)
#model = model.cuda()
sent = ['This is a young man.', 'The older doctor there is a black woman.']
inputs = tokenizer(sent, padding=True, return_tensors='pt')
#inputs = inputs.cuda()
labels = inputs['input_ids'].clone()
labels.masked_fill_(inputs['attention_mask'] == 0, -100)

print(model(**inputs, labels=labels).logits.shape)
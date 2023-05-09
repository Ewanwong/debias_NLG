import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
import regex as re
from tqdm import tqdm
from models.other_utils import get_intervals, load_file_to_list
import random
import numpy as np

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Initialize the GPT-2 model and tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token

def set_random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_lm_loss(model, tokenizer, sentences_pairs):
    
    # LM loss

    # tokenizer sentences
    inputs = tokenizer([tokenizer.bos_token+sent+tokenizer.eos_token for sent_pair in sentences_pairs for sent in sent_pair], padding=True, return_tensors='pt').to(model.device)
    # print(inputs)

    # Get the input IDs and labels
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)

    # Forward pass through the model
    outputs = model(**inputs, labels=labels) 
    loss = outputs[0]

    return loss

    

def construct_prefix_pairs(sentences_pairs, female_words, male_words, neutral_words):
    # construct pairs
    prefix_gender = []  # The doctor is a young man/woman.
    train_pairs_neutral = [[], []] # He is a doctor vs. She is a nurse
    prefix_gender_prior = [] # He/She 
    sents1, sents2 = sentences_pairs
    for sent1, sent2 in zip(sents1, sents2):
        
        sent1_tokens, sent2_tokens = [tok.strip().lower() for tok in re.findall(pat, sent1)], [tok.strip().lower() for tok in re.findall(pat, sent2)] # case sensitive?
        assert len(sent1_tokens) == len(sent2_tokens)

        intervals = get_intervals(sent1, sent1_tokens)
        
        gender_present = False
        neutral_present = False
        for i in range(len(sent1_tokens)):

            if not gender_present and not neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent_prefix = ''

                if i != 0:
                    for j in range(i):
                        sent_prefix += ' '*intervals[j] + sent1_tokens[j]
                        
                
                prefix_gender.append(sent_prefix)
                
                gender_present = True

            elif not gender_present and neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent_prefix = ''

                if i != 0:
                    for j in range(i):
                        sent_prefix += ' '*intervals[j] + sent1_tokens[j]
                        
                
                prefix_gender_prior.append(sent_prefix)
                
                gender_present = True

            if gender_present and sent1_tokens[i].lower() in neutral_words:
                sent1_prefix, sent2_prefix = '', ''

                if i != 0:
                    for j in range(i):
                        sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                        sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                
                train_pairs_neutral[0].append(sent1_prefix)
                train_pairs_neutral[1].append(sent2_prefix)
                neutral_present = True

    return prefix_gender, tuple(train_pairs_neutral), prefix_gender_prior

def get_gender_loss(model, tokenizer, prefix_gender, male_words, female_words, jsd_model):
    

    sents1_vocab_dist = get_conditional_prob_dist_fast(model, tokenizer, prefix_gender, male_words)
    sents2_vocab_dist = get_conditional_prob_dist_fast(model, tokenizer, prefix_gender, female_words)

    loss = jsd_model(sents1_vocab_dist, sents2_vocab_dist)

    return loss





def get_neutral_loss(model, tokenizer, train_pairs_neutral, neutral_words, jsd_model):
    sents1_prefix, sents2_prefix = train_pairs_neutral
    sents1_vocab_dist = get_conditional_prob_dist_fast(model, tokenizer, sents1_prefix, neutral_words)
    sents2_vocab_dist = get_conditional_prob_dist_fast(model, tokenizer, sents2_prefix, neutral_words)
    

    loss = jsd_model(sents1_vocab_dist, sents2_vocab_dist)

    return loss



def get_sent_prob_diff_loss(model, tokenizer, sentences_pairs):
    male_sents, female_sents = sentences_pairs
    male_inputs = tokenizer([tokenizer.bos_token+male_sent+tokenizer.eos_token for male_sent in male_sents], padding=True, return_tensors='pt').to(model.device)
    female_inputs = tokenizer([tokenizer.bos_token+female_sent+tokenizer.eos_token for female_sent in female_sents], padding=True, return_tensors='pt').to(model.device)
    
    male_input_ids = male_inputs["input_ids"]
    male_attention_mask = male_inputs["attention_mask"]
    male_labels = male_input_ids.clone()
    male_labels.masked_fill_(male_attention_mask == 0, -100)

    female_input_ids = female_inputs["input_ids"]
    female_attention_mask = female_inputs["attention_mask"]
    female_labels = female_input_ids.clone()
    female_labels.masked_fill_(female_attention_mask == 0, -100)

    male_logits = model(**male_inputs).logits
    female_logits = model(**female_inputs).logits

    shift_male_logits = male_logits[..., :-1, :].contiguous()
    shift_male_labels = male_labels[..., 1:].contiguous()

    shift_female_logits = female_logits[..., :-1, :].contiguous()
    shift_female_labels = female_labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction='mean')

    male_losses = []
    female_losses = []

    for i in range(shift_male_logits.shape[0]):

        male_loss = loss_fct(shift_male_logits[i,...].view(-1, shift_male_logits.size(-1)), shift_male_labels[i,...].view(-1))
        female_loss = loss_fct(shift_female_logits[i,...].view(-1, shift_female_logits.size(-1)), shift_female_labels[i,...].view(-1))
        male_losses.append(male_loss)
        female_losses.append(female_loss)
    
    male_prob = torch.exp(-torch.stack(male_losses))
    female_prob = torch.exp(-torch.stack(female_losses))

    sent_prob_diff_loss = 0.5 * torch.sum((male_prob - female_prob) ** 2) / len(male_sents)
    return sent_prob_diff_loss

class JSD(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_probs, net_2_probs):
        #net_1_probs = F.softmax(net_1_logits, dim=1)
        #net_2_probs= F.softmax(net_2_logits, dim=1)
        # batch * vocab_size
        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(torch.log(net_1_probs, dim=1), total_m, reduction=self.reduction) 
        loss += F.kl_div(torch.log(net_2_probs, dim=1), total_m, reduction=self.reduction) 
     
        return (0.5 * loss) 


    
def get_conditional_prob_dist_slow(model, tokenizer, prompts, word_list):

    prompts = [tokenizer.bos_token+sent for sent in prompts]
    
    # full = [tokenizer.bos_token+"I love doing aerobics", tokenizer.bos_token+'My favorite sport is aerobics']

    bsz = len(prompts)
    prompt_input = tokenizer(prompts, padding=True, return_tensors='pt')
    prompt_input = {k:v.to(model.device) for k,v in prompt_input.items()}
    # labels = prompt_input['input_ids'].clone()
    # labels.masked_fill_(prompt_input['attention_mask']==0, -100)
    prompt_output = model(**prompt_input, use_cache=True) # labels=labels
    past_key_values=prompt_output.past_key_values
    # loss1 = prompt_output.loss * (torch.sum(prompt_input['attention_mask']) - bsz)

    conditional_prob_distribution = []
    for word in word_list:
        continuation = [' '+word]
        continue_input = tokenizer(continuation, padding=True, return_tensors='pt')
        continue_input = {k:torch.stack((v,)*bsz).squeeze(-2).to(model.device) for k,v in continue_input.items()}
        labels = continue_input['input_ids'].clone()
        labels.masked_fill_(continue_input['attention_mask']==0, -100)
        
        continue_mask = torch.cat((prompt_input['attention_mask'], continue_input['attention_mask']), dim=1)
        continue_output = model(input_ids=continue_input['input_ids'], attention_mask=continue_mask, past_key_values=past_key_values)
    


        prompt_last_output_index = [torch.sum(prompt_input['attention_mask'][i,:]).item() for i in range(bsz)]
        prompt_last_output_logits = torch.stack([prompt_output.logits[i, prompt_last_output_index[i]-1, :] for i in range(bsz)]).unsqueeze(1)
        
        #continue_last_output_index = [torch.sum(continue_input['attention_mask'][i,:]).item() for i in range(bsz)]
        #continue_last_output_logits = torch.stack([continue_output.logits[i, :continue_last_output_index[i], :] for i in range(bsz)])
        

        logits = torch.cat((prompt_last_output_logits, continue_output.logits[:,:-1,:]), dim=1).contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.contiguous().view(-1)).view(bsz, -1)
        dist = torch.exp(-loss)
        conditional_prob_distribution.append(dist)
    
    conditional_prob_distribution = torch.stack(conditional_prob_distribution).view(bsz, -1) # bsz, word_num

    return conditional_prob_distribution / torch.sum(conditional_prob_distribution,dim=1).unsqueeze(1).expand(-1, conditional_prob_distribution.shape[1])


def get_conditional_prob_dist_fast(model, tokenizer, prompts, word_list):
    prompts = [tokenizer.bos_token+sent for sent in prompts]


    bsz = len(prompts)

    # encode prompt
    prompt_input = tokenizer(prompts, padding=True, return_tensors='pt')

    # take out the last token of each input (for conditional probability)
    mask_num = torch.sum(prompt_input['attention_mask'], dim=1)

    add_token_id = torch.tensor([prompt_input['input_ids'][i, mask_num[i].item()-1] for i in range(bsz)]).view(bsz,-1)
    prompt_input['input_ids'] = torch.stack([torch.cat((prompt_input['input_ids'][i, :mask_num[i].item()-1].unsqueeze(0), prompt_input['input_ids'][i, mask_num[i].item()-1:].unsqueeze(0)), dim=1) for i in range(bsz)]).squeeze(1)
    prompt_input['attention_mask'] = torch.stack([torch.cat((prompt_input['attention_mask'][i, :mask_num[i].item()-1].unsqueeze(0), prompt_input['attention_mask'][i, mask_num[i].item()-1:].unsqueeze(0)), dim=1) for i in range(bsz)]).squeeze(1)

    # process prompt
    prompt_input = {k:v.to(model.device) for k,v in prompt_input.items()}
    labels = prompt_input['input_ids'].clone()
    labels.masked_fill_(prompt_input['attention_mask']==0, -100)
    prompt_output = model(**prompt_input, use_cache=True) # labels=labels
    past_key_values=prompt_output.past_key_values
    # loss1 = prompt_output.loss * (torch.sum(prompt_input['attention_mask']) - bsz)

    conditional_prob_distribution = []
    for word in word_list:
        continuation = [' '+word] * bsz
        # encode continuation
        continue_input = tokenizer(continuation, padding=True, return_tensors='pt')

        # prepend the last token of the prompt to the continuation 
        add_mask = torch.ones((bsz, 1))
        continue_input['input_ids'] = torch.cat((add_token_id, continue_input['input_ids']), dim=1)
        continue_input['attention_mask'] = torch.cat((add_mask, continue_input['attention_mask']), dim=1)
        continue_input = {k:v.to(model.device) for k,v in continue_input.items()}
        labels = continue_input['input_ids'].clone()
        labels.masked_fill_(continue_input['attention_mask']==0, -100)

        # process attention mask for past_key_values
        continue_mask = torch.cat((prompt_input['attention_mask'], continue_input['attention_mask']), dim=1)
        continue_output = model(input_ids=continue_input['input_ids'], attention_mask=continue_mask, past_key_values=past_key_values)

        # calculate loss
        logits = continue_output.logits[..., :-1, :].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels[..., 1:].contiguous().contiguous().view(-1)).view(bsz, -1)

        loss = torch.sum(loss, dim=1)
        dist = torch.exp(-loss)
        conditional_prob_distribution.append(dist)
    
    conditional_prob_distribution = torch.stack(conditional_prob_distribution).view(bsz, -1) # bsz, word_num

    return conditional_prob_distribution / torch.sum(conditional_prob_distribution,dim=1).unsqueeze(1).expand(-1, conditional_prob_distribution.shape[1])


  

    

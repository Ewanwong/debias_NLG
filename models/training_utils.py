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
    inputs = tokenizer([tokenizer.bos_token+sent+tokenizer.eos_token for sent_pair in sentences_pairs for sent in sent_pair], padding=True, truncation=True, return_tensors='pt')
    inputs = {k:v.to('cuda:0') for k,v in inputs.items()}
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
    neutral_pairs = [[], []] # He is a doctor vs. She is a nurse
    prefix_gender_prior = [] # He/She 
    sents1, sents2 = sentences_pairs
    for sent1, sent2 in zip(sents1, sents2):
        
        sent1_tokens, sent2_tokens = [tok.strip().lower() for tok in re.findall(pat, sent1)], [tok.strip().lower() for tok in re.findall(pat, sent2)] # case sensitive?
        if len(sent1_tokens) != len(sent2_tokens):
            print(sent1)
            print(sent1_tokens)
            print(sent2)
            print(sent2_tokens)
            raise ValueError
        # assert len(sent1_tokens) == len(sent2_tokens)

        intervals = get_intervals(sent1, sent1_tokens)
        
        gender_present = False
        neutral_present = False
        for i in range(len(sent1_tokens)):

            if not gender_present and neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent_prefix = ''

                if i != 0:
                    for j in range(i):
                        sent_prefix += ' '*intervals[j] + sent1_tokens[j]
                        
                
                prefix_gender.append(sent_prefix)
                
                gender_present = True

            elif not gender_present and not neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent_prefix = ''

                if i != 0:
                    for j in range(i):
                        sent_prefix += ' '*intervals[j] + sent1_tokens[j]
                        
                
                prefix_gender_prior.append(sent_prefix)
                
                gender_present = True

            if sent1_tokens[i].lower() in neutral_words:
                if gender_present:
                    sent1_prefix, sent2_prefix = '', ''

                    if i != 0:
                        for j in range(i):
                            sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                            sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                    
                    neutral_pairs[0].append(sent1_prefix)
                    neutral_pairs[1].append(sent2_prefix)
                    neutral_present = True
                else:
                    neutral_present = True

    return prefix_gender, tuple(neutral_pairs), prefix_gender_prior

def get_gender_loss(model, tokenizer, prefix_gender, male_words, female_words, kld_model, batch_size):
    

    sents1_vocab_dist = get_conditional_prob_dist(model, tokenizer, prefix_gender, male_words, batch_size).view(-1)
    sents2_vocab_dist = get_conditional_prob_dist(model, tokenizer, prefix_gender, female_words, batch_size).view(-1)

    full_dist = torch.stack([sents1_vocab_dist, sents2_vocab_dist]).permute([1, 0])

    full_dist = full_dist / torch.sum(full_dist,dim=1).unsqueeze(1).expand(-1, full_dist.shape[1])

    loss = kld_model(full_dist)

    return loss





def get_neutral_loss(model, tokenizer, train_pairs_neutral, neutral_words, jsd_model, batch_size):
    sents1_prefix, sents2_prefix = train_pairs_neutral
    full_dist = get_conditional_prob_dist(model, tokenizer, sents1_prefix, neutral_words, batch_size)
    sents1_vocab_dist = full_dist / torch.sum(full_dist,dim=1).unsqueeze(1).expand(-1, full_dist.shape[1])
    full_dist = get_conditional_prob_dist(model, tokenizer, sents2_prefix, neutral_words, batch_size)
    sents2_vocab_dist = full_dist / torch.sum(full_dist,dim=1).unsqueeze(1).expand(-1, full_dist.shape[1])
    

    loss = jsd_model(sents1_vocab_dist, sents2_vocab_dist)

    return loss



def get_sent_prob_diff_loss(model, tokenizer, sentences_pairs, kld_model):
    male_sents, female_sents = sentences_pairs
    male_inputs = tokenizer([tokenizer.bos_token+male_sent+tokenizer.eos_token for male_sent in male_sents], padding=True, truncation=True, return_tensors='pt')
    male_inputs =  {k:v.to(model.device) for k,v in male_inputs.items()}
    female_inputs = tokenizer([tokenizer.bos_token+female_sent+tokenizer.eos_token for female_sent in female_sents], padding=True, truncation=True, return_tensors='pt').to(model.device)
    female_inputs =  {k:v.to(model.device) for k,v in female_inputs.items()}

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

    prob_dist = torch.stack([male_prob, female_prob]).permute([1, 0])

    prob_dist = prob_dist / torch.sum(prob_dist,dim=1).unsqueeze(1).expand(-1, prob_dist.shape[1])
 

    loss = kld_model(prob_dist)
    return loss

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
        loss += F.kl_div(torch.log(net_1_probs), total_m, reduction=self.reduction) 
        loss += F.kl_div(torch.log(net_2_probs), total_m, reduction=self.reduction) 
     
        return (0.5 * loss) 
    
class KLD(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(KLD, self).__init__()
        self.reduction = reduction

    def forward(self, predicted_probs):
        #net_1_probs = F.softmax(net_1_logits, dim=1)
        #net_2_probs= F.softmax(net_2_logits, dim=1)
        # batch * vocab_size
        assert predicted_probs.shape[1] == 2
        batch = predicted_probs.shape[0]
        target_probs = torch.ones((batch, 2)) * 0.5
        target_probs = target_probs.to(predicted_probs.device)
        loss = 0.0
        loss += F.kl_div(torch.log(predicted_probs), target_probs, reduction=self.reduction) 
     
        return (0.5 * loss) 





def get_conditional_prob_dist(model, tokenizer, prompts, word_list, batch_size):
    
    full_dist = []
    for i in range(int(len(prompts)/batch_size)+1):
        if i * batch_size >= len(prompts):
            break
        elif (i+1) * batch_size >= len(prompts):
            batch_prompts = prompts[i*batch_size:]
        else:
            batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
        batch_prompts = [tokenizer.bos_token+sent for sent in batch_prompts]

        bsz = len(batch_prompts)

        # encode prompt
        prompt_input = tokenizer(batch_prompts, padding=True, return_tensors='pt')

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
        
        conditional_prob_distribution = torch.stack(conditional_prob_distribution).permute([1, 0]) 
        full_dist.append(conditional_prob_distribution)
    full_dist = torch.cat(full_dist, dim=0)
    return full_dist



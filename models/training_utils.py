import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
import regex as re
from tqdm import tqdm
from other_utils import get_intervals, load_file_to_list
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
    train_pairs_gender = [[], []]  # The doctor is a young man/woman.
    train_pairs_neutral = [[], []] # He is a doctor vs. She is a nurse
    train_pairs_gender_prior = [[], []] # He/She 

    for sent1, sent2 in sentences_pairs:
        
        sent1_tokens, sent2_tokens = [tok.strip().lower() for tok in re.findall(pat, sent1)], [tok.strip().lower() for tok in re.findall(pat, sent2)] # case sensitive?
        assert len(sent1_tokens) == len(sent2_tokens)

        intervals = get_intervals(sent1, sent1_tokens)
        
        gender_present = False
        neutral_present = False
        for i in range(len(sent1_tokens)):

            if not gender_present and not neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent1_prefix, sent2_prefix = '', ''

                if i != 0:
                    for j in range(i):
                        sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                        sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                
                train_pairs_gender[0].append(sent1_prefix)
                train_pairs_gender[1].append(sent2_prefix)
                gender_present = True

            elif not gender_present and neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent1_prefix, sent2_prefix = '', ''

                if i != 0:
                    for j in range(i):
                        sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                        sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                
                train_pairs_gender_prior[0].append(sent1_prefix)
                train_pairs_gender_prior[1].append(sent2_prefix)
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

    return tuple(train_pairs_gender), tuple(train_pairs_neutral), tuple(train_pairs_gender_prior)

def get_gender_loss(model, tokenizer, train_pairs, fine_tuning_male_vocab, fine_tuning_female_vocab, jsd_model):
    sents1_prefix, sents2_prefix = train_pairs

    sents1_tokenized = tokenizer(sents1_prefix, padding=True, return_tensors='pt').to(model.device)
    sents2_tokenized = tokenizer(sents2_prefix, padding=True, return_tensors='pt').to(model.device)

    sents1_predictions = model(**sents1_tokenized)
    sents2_predictions = model(**sents2_tokenized)

    sents1_predictions_logits = sents1_predictions.logits[:, fine_tuning_male_vocab]
    sents2_predictions_logits = sents2_predictions.logits[:, fine_tuning_female_vocab]

    loss = jsd_model(sents1_predictions_logits, sents2_predictions_logits)

    return loss


def get_neutral_loss(model, tokenizer, train_pairs_neutral, fine_tuning_vocab, jsd_model):
    sents1_prefix, sents2_prefix = train_pairs_neutral

    sents1_tokenized = tokenizer(sents1_prefix, padding=True, return_tensors='pt').to(model.device)
    sents2_tokenized = tokenizer(sents2_prefix, padding=True, return_tensors='pt').to(model.device)

    sents1_predictions = model(**sents1_tokenized, use_cache=True)
    sents2_predictions = model(**sents2_tokenized, use_cache=True)

    sents1_past_key_values = sents1_predictions.past_key_values
    sents2_past_key_values = sents2_predictions.past_key_values # pass past_key_values to model to get conditional probability



def get_neutral_loss(model, tokenizer, train_pairs_neutral, fine_tuning_vocab, jsd_model):
    sents1_prefix, sents2_prefix = train_pairs_neutral

    sents1_tokenized = tokenizer(sents1_prefix, padding=True, return_tensors='pt').to(model.device)
    sents2_tokenized = tokenizer(sents2_prefix, padding=True, return_tensors='pt').to(model.device)

    sents1_predictions = model(**sents1_tokenized)
    sents2_predictions = model(**sents2_tokenized)

    sents1_predictions_logits = sents1_predictions.logits[:, fine_tuning_vocab]
    sents2_predictions_logits = sents2_predictions.logits[:, fine_tuning_vocab]

    loss = jsd_model(sents1_predictions_logits, sents2_predictions_logits)

    return loss

def get_fine_tuning_vocab(tokenizer, words_list):
    ids = []
    for word in words_list:
        id = tokenizer.encode(word, add_special_tokens=False)
        ids.append(id)
        
            
    return ids

def get_fine_tuning_gender_vocab(tokenizer, male_list, female_list):
    fine_tuning_male_vocab = get_fine_tuning_vocab(tokenizer, male_list)
    fine_tuning_female_vocab = get_fine_tuning_vocab(tokenizer, female_list)
    # assert len(fine_tuning_male_vocab) == len(fine_tuning_female_vocab)
    return fine_tuning_male_vocab, fine_tuning_female_vocab

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

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs= F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction=self.reduction) 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction=self.reduction) 
     
        return (0.5 * loss) 


    
def get_conditional_prob_dist(model, tokenizer, sents, word_list):

    prompt = [tokenizer.bos_token+sent for sent in sents]
    
    # full = [tokenizer.bos_token+"I love doing aerobics", tokenizer.bos_token+'My favorite sport is aerobics']

    bsz = len(prompt)
    prompt_input = tokenizer(prompt, padding=True, return_tensors='pt')
    prompt_input = {k:v.to(model.device) for k,v in prompt_input.items()}
    # labels = prompt_input['input_ids'].clone()
    # labels.masked_fill_(prompt_input['attention_mask']==0, -100)
    prompt_output = model(**prompt_input, use_cache=True) # labels=labels
    past_key_values=prompt_output.past_key_values
    # loss1 = prompt_output.loss * (torch.sum(prompt_input['attention_mask']) - bsz)
    """
    full_input = tokenizer(full, padding=True, return_tensors='pt')
    full_input = {k:v.cuda() for k,v in full_input.items()}
    labels = full_input['input_ids'].clone()
    labels.masked_fill_(full_input['attention_mask']==0, -100)
    full_output = model(**full_input, labels=labels)   
    loss2 = full_output.loss * (torch.sum(full_input['attention_mask']) - bsz)
    """
    for word in word_list:
        continuation = [' '+word]
        continue_input = tokenizer(continuation, padding=True, return_tensors='pt')
        continue_input = {k:torch.stack((v,)*bsz).to(model.device) for k,v in continue_input.items()}
        labels = continue_input['input_ids'].clone()
        labels.masked_fill_(continue_input['attention_mask']==0, -100)
        continue_mask = torch.cat((prompt_input['attention_mask'], continue_input['attention_mask']), dim=1)
        continue_output = model(input_ids=continue_input['input_ids'], attention_mask=continue_mask, past_key_values=past_key_values)
    
    """
    full_last_output_index = [torch.sum(full_input['attention_mask'][i,:]).item() for i in range(bsz)]
    full_last_output_logits = torch.stack([full_output.logits[i, full_last_output_index[i]-1, :] for i in range(bsz)]).unsqueeze(1)
    """

    prompt_last_output_index = [torch.sum(prompt_input['attention_mask'][i,:]).item() for i in range(bsz)]
    prompt_last_output_logits = torch.stack([prompt_output.logits[i, prompt_last_output_index[i]-1, :] for i in range(bsz)]).unsqueeze(1)
    
    continue_last_output_index = [torch.sum(continue_input['attention_mask'][i,:]).item() for i in range(bsz)]
    continue_last_output_logits = torch.stack([continue_output.logits[i, continue_last_output_index[i]-1, :] for i in range(bsz)]).unsqueeze(1)
    

    logits = torch.cat((prompt_last_output_logits, continue_last_output_logits), dim=1).contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.contiguous().view(-1))
    loss = torch.sum(loss, dim=1)
    print(loss)
    print(torch.exp(-loss))
  

    

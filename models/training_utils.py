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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lm_loss(model, tokenizer, sentences_pairs):
    # NLL on CDA data

    # LM loss

    # tokenizer sentences
    inputs = tokenizer([tokenizer.bos_token+sent+tokenizer.eos_token for sent_pair in sentences_pairs for sent in sent_pair], padding=True, truncation=True, return_tensors='pt')
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
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
        # print(sent1_tokens)
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

def get_gender_loss(model, tokenizer, prefix_gender, male_word_ids, female_word_ids, kld_model, batch_size, beta=10):
    #  kld for equalizing loss

    sents1_vocab_dist = get_conditional_prob_dist(model, tokenizer, prefix_gender, male_word_ids, batch_size).view(-1)
    sents2_vocab_dist = get_conditional_prob_dist(model, tokenizer, prefix_gender, female_word_ids, batch_size).view(-1)

    full_dist = torch.stack([sents1_vocab_dist, sents2_vocab_dist]).permute([1, 0])
    
    full_dist = F.softmax(full_dist/beta, dim=1)
   
    loss = kld_model(full_dist)
    
    return loss



def get_neutral_loss(model, tokenizer, train_pairs_neutral, neutral_word_ids, jsd_model, batch_size):
    # jsd for neutralization loss
    sents1_prefix, sents2_prefix = train_pairs_neutral
    sents1_dist = get_conditional_prob_dist(model, tokenizer, sents1_prefix, neutral_word_ids, batch_size)
    #sents1_vocab_dist = full_dist / torch.sum(full_dist,dim=1).unsqueeze(1).expand(-1, full_dist.shape[1])
    sents2_dist = get_conditional_prob_dist(model, tokenizer, sents2_prefix, neutral_word_ids, batch_size)
    #  sents2_vocab_dist = full_dist / torch.sum(full_dist,dim=1).unsqueeze(1).expand(-1, full_dist.shape[1])
    sents1_dist = F.softmax(sents1_dist, dim=1)
    sents2_dist = F.softmax(sents2_dist, dim=1)
    
    loss = jsd_model(sents1_dist, sents2_dist)

    return loss



def get_sent_prob_diff_loss(model, tokenizer, sentences_pairs, kld_model):
    # seq level equalizing loss
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
    

    prob_dist = torch.stack([torch.stack(male_losses), torch.stack(female_losses)]).permute([1, 0])

    prob_dist = F.softmax(prob_dist, dim=1)

    loss = kld_model(prob_dist)
    return loss

class JSD(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_probs, net_2_probs):
        # net_1_probs = F.softmax(net_1_logits, dim=1)
        # net_2_probs= F.softmax(net_2_logits, dim=1)
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

def get_vocab_id(tokenizer, word_list):
    eos = tokenizer.encode(tokenizer.eos_token)[0]

    id_list = []
    for word in word_list:
        id = tokenizer.encode(word)[0]
        if  id != eos:
            id_list.append(id)
    return id_list



def get_conditional_prob_dist(model, tokenizer, prompts, id_list, batch_size):
    # next token probability
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

        

        # process prompt
        prompt_input = {k:v.to(model.device) for k,v in prompt_input.items()}
        
        prompt_output = model(**prompt_input) # labels=labels
        

        conditional_prob_distribution = prompt_output.logits[:, -1, id_list] #bsz word_num
 
        full_dist.append(conditional_prob_distribution)
    full_dist = torch.cat(full_dist, dim=0)

    return full_dist
    
def clean_vocab(tokenizer, male_words, female_words, neutral_words):
    unk = tokenizer.encode(tokenizer.unk_token)[0]
    new_male_words, new_female_words, new_neutral_words = [], [], []
    for m, f in zip(male_words, female_words):
        m_id, f_id = tokenizer.encode(m), tokenizer.encode(f)
        if len(m_id) < 2 and m_id[0] != unk and len(f_id) < 2 and f_id[0] !=unk:
            new_male_words.append(m)
            new_female_words.append(f)
    for n in neutral_words:
        n_id = tokenizer.encode(n)
        if len(n_id) < 2 and n_id[0] != unk:
            new_neutral_words.append(n)
    return new_male_words, new_female_words, new_neutral_words

from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from transformers import get_scheduler
from models.PrefixGPT2 import PrefixGPT2
from models.other_utils import load_file_to_list
from models.training_utils import get_lm_loss, get_gender_loss, get_neutral_loss, get_sent_prob_diff_loss, construct_prefix_pairs, get_fine_tuning_vocab, get_fine_tuning_gender_vocab, JSD
import json
from torch.optim import AdamW
from tqdm.auto import tqdm

class DebiasedDataset(Dataset):
    def __init__(self, data_path, gender_swapped_data_path) -> None:
        super().__init__()
        self.data = load_file_to_list(data_path)
        self.gender_swapped_data = load_file_to_list(gender_swapped_data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.gender_swapped_data[index]
    
def main():

    with open("config.json", 'r') as f:
        config = json.load(f) # gpt2 config + training config + prefix model config


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = DebiasedDataset(config.train_data_path, config.gender_swapped_data_path)
    dev_dataset = DebiasedDataset(config.dev_data_path, config.dev_gender_swapped_data_path)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.bsz)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.bsz)

    # initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    male_words = load_file_to_list('data/male.txt')
    female_words = load_file_to_list('data/female.txt')
    neutral_words = load_file_to_list('data/neutral.txt')
    fine_tuning_male_vocab, fine_tuning_female_vocab = get_fine_tuning_gender_vocab(tokenizer, male_words, female_words)
    fine_tuning_neutral_vocab = get_fine_tuning_vocab(tokenizer, neutral_words)

    # initialize prefix model
    model = PrefixGPT2(config) 

    # initialize JSD model
    jsd_model = JSD()

    # initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr)
    num_epochs = config.epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_training_steps
    )

    model = model.to(device)

    # start training
    progress_bar = tqdm(range(num_training_steps))


    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            train_pairs_gender, train_pairs_neutral, train_pairs_gender_prior = construct_prefix_pairs(batch, female_words, male_words, neutral_words)
            # calculate losses
            lm_loss = get_lm_loss(model, tokenizer, batch)
            gender_loss = get_gender_loss(model, tokenizer, train_pairs_gender, fine_tuning_male_vocab, fine_tuning_female_vocab, jsd_model)
            neutral_loss = get_neutral_loss(model, tokenizer, train_pairs_neutral, fine_tuning_neutral_vocab, jsd_model)
            gender_prior_loss = get_gender_loss(model, tokenizer, train_pairs_gender_prior, fine_tuning_male_vocab, fine_tuning_female_vocab, jsd_model)
            sent_prob_diff_loss = get_sent_prob_diff_loss(model, tokenizer, batch)

            loss = lm_loss + config.alpha1 * gender_loss + config.alpha2 * neutral_loss + config.alpha3 * gender_prior_loss + config.alpha4 * sent_prob_diff_loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            #TODO: save checkpoint
        
        # evaluation
        model.eval()
        losses = []
        for batch in dev_dataloader:
            train_pairs_gender, train_pairs_neutral, train_pairs_gender_prior = construct_prefix_pairs(batch, female_words, male_words, neutral_words)
            # calculate losses
            lm_loss = get_lm_loss(model, tokenizer, batch)
            gender_loss = get_gender_loss(model, tokenizer, train_pairs_gender, fine_tuning_male_vocab, fine_tuning_female_vocab, jsd_model)
            neutral_loss = get_neutral_loss(model, tokenizer, train_pairs_neutral, fine_tuning_neutral_vocab, jsd_model)
            gender_prior_loss = get_gender_loss(model, tokenizer, train_pairs_gender_prior, fine_tuning_male_vocab, fine_tuning_female_vocab, jsd_model)

            loss = lm_loss + config.alpha1 * gender_loss + config.alpha2 * neutral_loss + config.alpha3 * gender_prior_loss
            losses.append(loss.item)
        dev_loss = sum(losses) / len(losses)
        # TODO: early stopping


from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AutoConfig, GPT2LMHeadModel
from transformers import get_scheduler
from models.PrefixGPT2 import PrefixGPT2
from models.other_utils import load_file_to_list, train_dev_split
from models.training_utils import get_lm_loss, get_gender_loss, get_neutral_loss, get_sent_prob_diff_loss, construct_prefix_pairs, JSD, set_random_seed
import json
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
import argparse

class DebiasedDataset(Dataset):
    def __init__(self, data, gender_swapped_data) -> None:
        super().__init__()
        self.data = data
        self.gender_swapped_data = gender_swapped_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.gender_swapped_data[index]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=['prefix_tuning', 'fine_tuning'],
        default='prefix_tuning')
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default='data/',
        help="Directory where all persistent data will be stored.")
    
    # training args
    parser.add_argument(
        "--lr",       
        default=1e-5
    )
    
    parser.add_argument(
        "--epochs",       
        default=5
    )
    
    parser.add_argument(
        "--batch_size",       
        default=8
    )
    
    parser.add_argument(
        "--warmup_steps",       
        default=1000
    )
    
    # model args
    parser.add_argument(
        "--pre_seq_len", 
        type=int,      
        default=10
    )
    
    parser.add_argument(
        "--prefix_projection",  
        type=bool,     
        default=True
    )
    
    parser.add_argument(
        "--prefix_hidden_size",       
        default=800
    )
    
    # loss args
    parser.add_argument(
        "--alpha1",       
        default=0.5
    )
    
    parser.add_argument(
        "--alpha2",       
        default=0.5
    )
    
    parser.add_argument(
        "--alpha3",       
        default=0.5
    )
    
    parser.add_argument(
        "--alpha4",       
        default=0.5
    )
    
    parser.add_argument(
        "--random_seed",
        default=42
    )
    
    args = parser.parse_args()

    # load & update config
    config = AutoConfig.from_pretrained('gpt2')
    config.update(vars(args))

    save_folder = f'alpha1_{config.alpha1}_alpha2_{config.alpha2}_alpha3_{config.alpha3}_alpha4_{config.alpha4}_pre_seq_len_{config.pre_seq_len}_prefix_projection_{config.prefix_projection}_hidden_size_{config.prefix_hidden_size}_lr_{config.lr}_batch_size_{config.batch_size}_warmup_{config.warmup_steps}'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    config.save_pretrained(os.path.join(save_folder, 'experiment_config.json'))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(config.random_seed)

    data_path = os.path.join(config.data_dir, 'data.txt')
    gender_swapped_data_path = os.path.join(config.data_dir, 'gender_swapped_data.txt')
    train_data, train_gender_swapped_data, dev_data, dev_gender_swapped_data = train_dev_split(data_path, gender_swapped_data_path, ratio=0.9, shuffle=True)


    # construct dataset/dataloader
    train_dataset = DebiasedDataset(train_data, train_gender_swapped_data)
    dev_dataset = DebiasedDataset(dev_data, dev_gender_swapped_data)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size)

    # initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    male_words = load_file_to_list('data/male.txt')
    female_words = load_file_to_list('data/female.txt')
    neutral_words = load_file_to_list('data/neutral.txt')

    # initialize prefix model
    if config.type == "prefix_tuning":
        model = PrefixGPT2(config) 
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')

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

    
    log_file = os.path.join(save_folder, 'log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        print(config, file=f)
        print("training starts", file=f)
        print(config)
        print("training starts")
        
        for epoch in range(num_epochs):
            print('==================================================================================', file=f)
            print(f"Epoch {epoch+1} starts", file=f)
            print('==================================================================================')
            print(f"Epoch {epoch+1} starts")
            epoch_save_path = os.path.join(save_folder, f'epoch_{epoch+1}')

            # collect train loss
            train_total_losses = []
            train_lm_losses = []
            train_gender_losses = []
            train_neutral_losses = []
            train_gender_prior_losses = []
            train_sent_prob_diff_losses = []

            model.train()
            for batch in train_dataloader:
                prefix_gender, train_pairs_neutral, prefix_gender_prior = construct_prefix_pairs(batch, female_words, male_words, neutral_words)
                # calculate losses
                lm_loss = get_lm_loss(model, tokenizer, batch)
                gender_loss = get_gender_loss(model, tokenizer, prefix_gender, male_words, female_words, jsd_model)
                neutral_loss = get_neutral_loss(model, tokenizer, train_pairs_neutral, neutral_words, jsd_model)
                gender_prior_loss = get_gender_loss(model, tokenizer, prefix_gender_prior, male_words, female_words, jsd_model)
                sent_prob_diff_loss = get_sent_prob_diff_loss(model, tokenizer, batch)

                loss = lm_loss + config.alpha1 * gender_loss + config.alpha2 * neutral_loss + config.alpha3 * gender_prior_loss + config.alpha4 * sent_prob_diff_loss
                train_total_losses.append(loss.item())
                train_lm_losses.append(lm_loss.item())
                train_gender_losses.append(gender_loss.item())
                train_neutral_losses.append(neutral_loss.item())
                train_gender_prior_losses.append(gender_prior_loss.item())
                train_sent_prob_diff_losses.append(sent_prob_diff_loss.item())
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # save model

            model.save_pretrained(epoch_save_path)

            print(f"train total loss: {sum(train_total_losses)/len(train_total_losses)}", file=f)
            print(f"train lm loss: {sum(train_lm_losses)/len(train_lm_losses)}", file=f)
            print(f"train gender loss: {sum(train_gender_losses)/len(train_gender_losses)}", file=f)
            print(f"train neutral loss: {sum(train_neutral_losses)/len(train_neutral_losses)}", file=f)
            print(f"train gender prior loss: {sum(train_gender_prior_losses)/len(train_gender_prior_losses)}", file=f)
            print(f"train sent prob diff loss: {sum(train_sent_prob_diff_losses)/len(train_sent_prob_diff_losses)}", file=f)

            print(f"train total loss: {sum(train_total_losses)/len(train_total_losses)}")
            print(f"train lm loss: {sum(train_lm_losses)/len(train_lm_losses)}")
            print(f"train gender loss: {sum(train_gender_losses)/len(train_gender_losses)}")
            print(f"train neutral loss: {sum(train_neutral_losses)/len(train_neutral_losses)}")
            print(f"train gender prior loss: {sum(train_gender_prior_losses)/len(train_gender_prior_losses)}")
            print(f"train sent prob diff loss: {sum(train_sent_prob_diff_losses)/len(train_sent_prob_diff_losses)}")

            
            # evaluation
            model.eval()
            
            # collect dev loss
            dev_total_losses = []
            dev_lm_losses = []
            dev_gender_losses = []
            dev_neutral_losses = []
            dev_gender_prior_losses = []
            dev_sent_prob_diff_losses = []

            for batch in dev_dataloader:
                dev_pairs_gender, dev_pairs_neutral, dev_pairs_gender_prior = construct_prefix_pairs(batch, female_words, male_words, neutral_words)
                # calculate losses
                lm_loss = get_lm_loss(model, tokenizer, batch)
                gender_loss = get_gender_loss(model, tokenizer, dev_pairs_gender, male_words, female_words, jsd_model)
                neutral_loss = get_neutral_loss(model, tokenizer, dev_pairs_neutral, neutral_words, jsd_model)
                gender_prior_loss = get_gender_loss(model, tokenizer, dev_pairs_gender_prior, male_words, female_words, jsd_model)

                loss = lm_loss + config.alpha1 * gender_loss + config.alpha2 * neutral_loss + config.alpha3 * gender_prior_loss

                dev_total_losses.append(loss.item())
                dev_lm_losses.append(lm_loss.item())
                dev_gender_losses.append(gender_loss.item())
                dev_neutral_losses.append(neutral_loss.item())
                dev_gender_prior_losses.append(gender_prior_loss.item())
                dev_sent_prob_diff_losses.append(sent_prob_diff_loss.item())

            print(f"dev total loss: {sum(dev_total_losses)/len(dev_total_losses)}", file=f)
            print(f"dev lm loss: {sum(dev_lm_losses)/len(dev_lm_losses)}", file=f)
            print(f"dev gender loss: {sum(dev_gender_losses)/len(dev_gender_losses)}", file=f)
            print(f"dev neutral loss: {sum(dev_neutral_losses)/len(dev_neutral_losses)}", file=f)
            print(f"dev gender prior loss: {sum(dev_gender_prior_losses)/len(dev_gender_prior_losses)}", file=f)
            print(f"dev sent prob diff loss: {sum(dev_sent_prob_diff_losses)/len(dev_sent_prob_diff_losses)}", file=f)

            print(f"dev total loss: {sum(dev_total_losses)/len(dev_total_losses)}")
            print(f"dev lm loss: {sum(dev_lm_losses)/len(dev_lm_losses)}")
            print(f"dev gender loss: {sum(dev_gender_losses)/len(dev_gender_losses)}")
            print(f"dev neutral loss: {sum(dev_neutral_losses)/len(dev_neutral_losses)}")
            print(f"dev gender prior loss: {sum(dev_gender_prior_losses)/len(dev_gender_prior_losses)}")
            print(f"dev sent prob diff loss: {sum(dev_sent_prob_diff_losses)/len(dev_sent_prob_diff_losses)}")

        print("training stops", file=f)
        print("training_stops")


if __name__  == "__main__":
    main()

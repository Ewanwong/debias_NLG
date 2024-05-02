from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AutoConfig, GPT2LMHeadModel
from transformers import get_scheduler
from models.PrefixGPT2 import PrefixGPT2
from models.other_utils import load_file_to_list, train_dev_split
from models.training_utils import get_lm_loss, get_gender_loss, get_neutral_loss, get_sent_prob_diff_loss, construct_prefix_pairs, JSD, KLD, set_random_seed, get_vocab_id, clean_vocab
import json
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
import argparse
from torch.nn import DataParallel

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
        default=5e-5
    )

    # lm for further pre-training
    parser.add_argument(
        "--lm_lr",       
        default=5e-5
    )
    
    parser.add_argument(
        "--lm_epochs",
        type=int,       
        default=0
    )

    parser.add_argument(
        "--lm_warmup_steps",
        type=int,       
        default=500
    )

    parser.add_argument(
        "--epochs",
        type=int,       
        default=5
    )
    
    parser.add_argument(
        "--lm_batch_size",
        type=int,       
        default=16
    )

    parser.add_argument(
        "--batch_size",   
        type=int,    
        default=16
    )
    
    parser.add_argument(
        "--warmup_steps",    
        type=int,   
        default=500
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
        type=int,       
        default=800
    )

    # loss args
    parser.add_argument(
        "--alpha1", 
        type=int,      
        default=1
    )
    
    parser.add_argument(
        "--alpha2", 
        type=int,      
        default=50
    )
    
    parser.add_argument(
        "--alpha3", 
        type=int,      
        default=200
    )
    
    parser.add_argument(
        "--alpha4", 
        type=int,      
        default=250
    )

    # alpha5 is discarded
    parser.add_argument(
        "--alpha5", 
        type=int,      
        default=0                  
    )

    parser.add_argument(
        "--beta",
        type=int,       
        default=10
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--use_full_words",
        default=True
    )
    
    args = parser.parse_args()


    # load & update config
    config = AutoConfig.from_pretrained('gpt2')
    config.update(vars(args))

    save_folder = f'fast_model_{config.type}_alpha1_{config.alpha1}_alpha2_{config.alpha2}_alpha3_{config.alpha3}_alpha4_{config.alpha4}_beta_{config.beta}_pre_seq_len_{config.pre_seq_len}_hidden_size_{config.prefix_hidden_size}_batch_size_{config.batch_size}_lm_batch_size_{config.lm_batch_size}_lr_{config.lr}_lm_lr_{config.lm_lr}_epochs_{config.epochs}_lm_epochs_{config.lm_epochs}_warmup_{config.warmup_steps}_lm_warmup_{config.lm_warmup_steps}_random_seed_{config.random_seed}'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    config.save_pretrained(save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_random_seed(config.random_seed)

    data_path = os.path.join(config.data_dir, 'data.txt')
    gender_swapped_data_path = os.path.join(config.data_dir, 'gender_swapped_data.txt')
    train_data, train_gender_swapped_data, dev_data, dev_gender_swapped_data = train_dev_split(data_path, gender_swapped_data_path, ratio=0.9, shuffle=True)


    # construct dataset/dataloader
    train_dataset = DebiasedDataset(train_data, train_gender_swapped_data)
    dev_dataset = DebiasedDataset(dev_data, dev_gender_swapped_data)

    # initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    male_words = load_file_to_list('data/male.txt')
    female_words = load_file_to_list('data/female.txt')
    neutral_words = list(set(load_file_to_list('data/neutral.txt')))

    if config.use_full_words:
        male_words, female_words, neutral_words = clean_vocab(tokenizer, male_words, female_words, neutral_words)

    male_ids = get_vocab_id(tokenizer, male_words)
    female_ids = get_vocab_id(tokenizer, female_words)
    neutral_ids = get_vocab_id(tokenizer, neutral_words)
        
    # initialize prefix model
    if config.type == "prefix_tuning":
        model = PrefixGPT2(config) 
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    # initialize JSD/KLD model
    jsd_model = JSD()
    kld_model = KLD()
    
    # initialize optimizer and scheduler
    model.to(device)
    """
    if device_count > 1:
        model = DataParallel(model, device_ids=devices)
    """
    # start training
    

    
    log_file = os.path.join(save_folder, 'log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        print(config, file=f)
        print("training starts", file=f)
        print(config)
        print("training starts")

        # lm tuning

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.lm_batch_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.lm_batch_size)

        lm_optimizer = AdamW(model.parameters(), lr=config.lm_lr)
        
        lm_epochs = config.lm_epochs

        lm_num_training_steps = lm_epochs * len(train_dataloader)
        
        lm_lr_scheduler = get_scheduler(
            name="linear", optimizer=lm_optimizer, num_warmup_steps=config.lm_warmup_steps, num_training_steps=lm_num_training_steps
        )
        
        lm_progress_bar = tqdm(range(lm_num_training_steps))
        

        for epoch in range(lm_epochs):
            print('==================================================================================', file=f)
            print(f"LM tuning Epoch {epoch+1} starts", file=f)
            print('==================================================================================')
            print(f"LM tuning Epoch {epoch+1} starts")
            epoch_save_path = os.path.join(save_folder, f'lm_epoch_{epoch+1}')

            # collect train loss
            train_total_losses = []
            train_lm_losses = []
            
            model.train()
            for batch in train_dataloader:
                
                 # calculate losses
                lm_loss = get_lm_loss(model, tokenizer, batch)

                loss = lm_loss


                train_total_losses.append(loss.item())
                train_lm_losses.append(lm_loss.item())
                loss.backward()

                lm_optimizer.step()
                lm_lr_scheduler.step()
                lm_optimizer.zero_grad()
                lm_progress_bar.update(1)

            # save model

            model.save_pretrained(epoch_save_path)

            print(f"train total loss: {sum(train_total_losses)/len(train_total_losses)}", file=f)
            print(f"train lm loss: {sum(train_lm_losses)/len(train_lm_losses)}", file=f)
            
            print(f"train total loss: {sum(train_total_losses)/len(train_total_losses)}")
            print(f"train lm loss: {sum(train_lm_losses)/len(train_lm_losses)}")
            
            
            # evaluation
            model.eval()
            with torch.no_grad():
        
                # collect dev loss
                dev_total_losses = []
                dev_lm_losses = []
                
                for batch in dev_dataloader:
                    
                    # calculate losses
                    lm_loss = get_lm_loss(model, tokenizer, batch)                    
                    loss = lm_loss
                    dev_total_losses.append(loss.item())
                    dev_lm_losses.append(lm_loss.item())
                    

            print(f"dev total loss: {sum(dev_total_losses)/len(dev_total_losses)}", file=f)
            print(f"dev lm loss: {sum(dev_lm_losses)/len(dev_lm_losses)}", file=f)
            

            print(f"dev total loss: {sum(dev_total_losses)/len(dev_total_losses)}")
            print(f"dev lm loss: {sum(dev_lm_losses)/len(dev_lm_losses)}")
            
        # debiasing training
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size)

        optimizer = AdamW(model.parameters(), lr=config.lr)

        num_epochs = config.epochs
        
        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))


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
            train_bias_losses = []

            model.train()
            for batch in train_dataloader:
                prefix_gender, neutral_pairs, prefix_gender_prior = construct_prefix_pairs(batch, female_words, male_words, neutral_words)
                
                 # calculate losses
                lm_loss = get_lm_loss(model, tokenizer, batch)
                
                
                if len(prefix_gender) > 0 and config.alpha2!=0:
                    gender_loss = get_gender_loss(model, tokenizer, prefix_gender, male_ids, female_ids, kld_model, config.batch_size, beta=config.beta)  # most time-consuming
                else:
                    gender_loss = torch.tensor(0.0)
                
                if len(neutral_pairs[0]) > 0 and config.alpha3!=0:
                    neutral_loss = get_neutral_loss(model, tokenizer, neutral_pairs, neutral_ids, jsd_model, config.batch_size)
                else:
                    neutral_loss = torch.tensor(0.0)
                
                if len(prefix_gender_prior) > 0 and config.alpha5!=0:
                    gender_prior_loss = get_gender_loss(model, tokenizer, prefix_gender_prior, male_ids, female_ids, kld_model, config.batch_size, beta=config.beta)
                else:
                    gender_prior_loss = torch.tensor(0.0)
                               
                if config.alpha4 != 0:
                    sent_prob_diff_loss = get_sent_prob_diff_loss(model, tokenizer, batch, kld_model)
                else:
                    sent_prob_diff_loss = torch.tensor(0.0)
                
                
                # print(lm_loss.item(), config.alpha2 * gender_loss.item(), config.alpha3 * neutral_loss.item(), config.alpha5 * gender_prior_loss.item(), config.alpha4 * sent_prob_diff_loss.item()) 
                
                loss = config.alpha1 * lm_loss + + config.alpha2 * neutral_loss + config.alpha3 * gender_loss  + config.alpha4 * sent_prob_diff_loss + config.alpha5 * gender_prior_loss 
                if torch.isnan(loss):
                    return

                train_total_losses.append(loss.item())
                train_lm_losses.append(lm_loss.item())
                train_gender_losses.append(gender_loss.item())
                train_neutral_losses.append(neutral_loss.item())
                train_gender_prior_losses.append(gender_prior_loss.item())
                train_sent_prob_diff_losses.append(sent_prob_diff_loss.item())
                train_bias_losses.append((gender_loss+neutral_loss+gender_prior_loss+sent_prob_diff_loss).item())
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
            print(f"train other loss: {sum(train_total_losses)/len(train_total_losses)-sum(train_lm_losses)/len(train_lm_losses)}", file=f)
            print(f"train bias loss: {sum(train_bias_losses)/len(train_bias_losses)}", file=f)

            print(f"train total loss: {sum(train_total_losses)/len(train_total_losses)}")
            print(f"train lm loss: {sum(train_lm_losses)/len(train_lm_losses)}")
            print(f"train gender loss: {sum(train_gender_losses)/len(train_gender_losses)}")
            print(f"train neutral loss: {sum(train_neutral_losses)/len(train_neutral_losses)}")
            print(f"train gender prior loss: {sum(train_gender_prior_losses)/len(train_gender_prior_losses)}")
            print(f"train sent prob diff loss: {sum(train_sent_prob_diff_losses)/len(train_sent_prob_diff_losses)}")
            print(f"train other loss: {sum(train_total_losses)/len(train_total_losses)-sum(train_lm_losses)/len(train_lm_losses)}")
            print(f"train bias loss: {sum(train_bias_losses)/len(train_bias_losses)}")

            
            # evaluation
            model.eval()
            with torch.no_grad():
        
                # collect dev loss
                dev_total_losses = []
                dev_lm_losses = []
                dev_gender_losses = []
                dev_neutral_losses = []
                dev_gender_prior_losses = []
                dev_sent_prob_diff_losses = []
                dev_bias_losses = []

                for batch in dev_dataloader:
                    prefix_gender, neutral_pairs, prefix_gender_prior = construct_prefix_pairs(batch, female_words, male_words, neutral_words)
                    # calculate losses
                    lm_loss = get_lm_loss(model, tokenizer, batch)
                    
                
                    if len(prefix_gender) > 0 and config.alpha2!=0:
                        gender_loss = get_gender_loss(model, tokenizer, prefix_gender, male_ids, female_ids, kld_model, config.batch_size, beta=config.beta)  # most time-consuming
                    else:
                        gender_loss = torch.tensor(0.0)
                    
                    if len(neutral_pairs[0]) > 0 and config.alpha3!=0:
                        neutral_loss = get_neutral_loss(model, tokenizer, neutral_pairs, neutral_ids, jsd_model, config.batch_size)
                    else:
                        neutral_loss = torch.tensor(0.0)
                    
                    if len(prefix_gender_prior) > 0 and config.alpha5!=0:
                        gender_prior_loss = get_gender_loss(model, tokenizer, prefix_gender_prior, male_ids, female_ids, kld_model, config.batch_size, beta=config.beta)
                    else:
                        gender_prior_loss = torch.tensor(0.0)
                                
                    if config.alpha4 != 0:
                        sent_prob_diff_loss = get_sent_prob_diff_loss(model, tokenizer, batch, kld_model)
                    else:
                        sent_prob_diff_loss = torch.tensor(0.0)
                    loss = config.alpha1 * lm_loss + config.alpha2 * neutral_loss + config.alpha3 * gender_loss + config.alpha4 * sent_prob_diff_loss + config.alpha5 * gender_prior_loss

                    dev_total_losses.append(loss.item())
                    dev_lm_losses.append(lm_loss.item())
                    dev_gender_losses.append(gender_loss.item())
                    dev_neutral_losses.append(neutral_loss.item())
                    dev_gender_prior_losses.append(gender_prior_loss.item())
                    dev_sent_prob_diff_losses.append(sent_prob_diff_loss.item())
                    dev_bias_losses.append((gender_loss+neutral_loss+gender_prior_loss+sent_prob_diff_loss).item())
            print(f"dev total loss: {sum(dev_total_losses)/len(dev_total_losses)}", file=f)
            print(f"dev lm loss: {sum(dev_lm_losses)/len(dev_lm_losses)}", file=f)
            print(f"dev gender loss: {sum(dev_gender_losses)/len(dev_gender_losses)}", file=f)
            print(f"dev neutral loss: {sum(dev_neutral_losses)/len(dev_neutral_losses)}", file=f)
            print(f"dev gender prior loss: {sum(dev_gender_prior_losses)/len(dev_gender_prior_losses)}", file=f)
            print(f"dev sent prob diff loss: {sum(dev_sent_prob_diff_losses)/len(dev_sent_prob_diff_losses)}", file=f)
            print(f"dev other loss: {sum(dev_total_losses)/len(dev_total_losses)-sum(dev_lm_losses)/len(dev_lm_losses)}", file=f)
            print(f"dev bias loss: {sum(dev_bias_losses)/len(dev_bias_losses)}", file=f)

            print(f"dev total loss: {sum(dev_total_losses)/len(dev_total_losses)}")
            print(f"dev lm loss: {sum(dev_lm_losses)/len(dev_lm_losses)}")
            print(f"dev gender loss: {sum(dev_gender_losses)/len(dev_gender_losses)}")
            print(f"dev neutral loss: {sum(dev_neutral_losses)/len(dev_neutral_losses)}")
            print(f"dev gender prior loss: {sum(dev_gender_prior_losses)/len(dev_gender_prior_losses)}")
            print(f"dev sent prob diff loss: {sum(dev_sent_prob_diff_losses)/len(dev_sent_prob_diff_losses)}")
            print(f"dev other loss: {sum(dev_total_losses)/len(dev_total_losses)-sum(dev_lm_losses)/len(dev_lm_losses)}")
            print(f"dev bias loss: {sum(dev_bias_losses)/len(dev_bias_losses)}")
        print("training stops", file=f)
        print("training_stops")


if __name__  == "__main__":
    main()

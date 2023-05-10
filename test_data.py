import argparse
import nltk
import regex as re
from tqdm import tqdm
from collections import defaultdict
import random
from models.other_utils import load_file_to_list, get_intervals

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

male_words = load_file_to_list('data/male.txt')
female_words = load_file_to_list('data/female.txt')

i= 0
for m, w in zip(male_words, female_words):
    if len(tokenizer.encode(m)) == 1 and len(tokenizer.encode(w))==1:
        i += 1
        
        print(m, w)
    
print(len(male_words))
print(i)

neutral_words = load_file_to_list('data/neutral.txt')
i= 0
for m in neutral_words:
    if len(tokenizer.encode(m)) == 1:
        i += 1
        
        print(m)
    
print(len(neutral_words))
print(i)
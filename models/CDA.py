import argparse
import nltk
import regex as re
from tqdm import tqdm
from collections import defaultdict
import random
from models.other_utils import load_file_to_list, get_intervals


def main():

    random.seed(1)



    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    input_path = 'data/data.txt'
    output_path = 'data/gender_swapped_data.txt'
    
    male_words = load_file_to_list('data/male.txt')
    female_words = load_file_to_list('data/female.txt')

    assert len(male_words) == len(female_words)
    male2female, female2male = defaultdict(list), defaultdict(list)
    for male_word, female_word in zip(male_words, female_words):
        
        male2female[male_word].append(female_word)
        female2male[female_word].append(male_word)

    
    sents = load_file_to_list(input_path)


    # by frequency
    count = {}
    all_tokens = [tok.strip().lower() for sent in sents for tok in re.findall(pat, sent)]
    for attribute in list(set(male_words + female_words)):
        count[attribute] = all_tokens.count(attribute) + 1 # smoothing
    male_weights = {}
    for male_word in male_words:
        female_words_list = male2female[male_word]
        weights = [count[i] for i in female_words_list]
        add =  sum(weights)
        weights = [count/add for count in weights]
        male_weights[male_word] = weights
    
    female_weights = {}
    for female_word in female_words:
        male_words_list = female2male[female_word]
        weights = [count[i] for i in male_words_list]
        add =  sum(weights)
        weights = [count/add for count in weights]
        female_weights[female_word] = weights
    

    # print(count)

    gender_swapped_sents = []
    for sent in tqdm(sents):
        toks = [tok.strip() for tok in re.findall(pat, sent)]
        # this is to make sure the transformed text has the same tokenization as original text
        intervals = get_intervals(sent, toks)
        
        swapped_sent = []
        for tok in toks:
            if tok.lower() in male_words:
                swapped_list = male2female[tok.lower()]
                
                swapped = random.choices(swapped_list, weights=male_weights[tok.lower()], k=1)[0]

                if tok[0] != tok.lower()[0]:
                    swapped = swapped.capitalize()
                swapped_sent.append(swapped)
            elif tok.lower() in female_words:
                swapped_list = female2male[tok.lower()]

                swapped = random.choices(swapped_list, weights=female_weights[tok.lower()], k=1)[0]
                
                if tok[0] != tok.lower()[0]:
                    swapped = swapped.capitalize()
                swapped_sent.append(swapped)
            else:
                swapped_sent.append(tok)
        assert len(toks) == len(swapped_sent)

        # add white spaces
        new_sent = ''
        for i in range(len(toks)):
            new_sent = new_sent + ' '*intervals[i] + swapped_sent[i]
        gender_swapped_sents.append(new_sent)
        # gender_swapped_sents.append(' '.join(swapped_sent))

    #TODO: add name swapping
    
    with open(output_path, 'w') as f:
        for sent in gender_swapped_sents:
            f.write(sent)
            f.write('\n')

if __name__ == '__main__':
    main()
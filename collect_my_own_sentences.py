import argparse
import regex as re
import nltk
import torch
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    tp = lambda x:list(x.split(','))

    parser.add_argument('--input', type=str, required=True,
                        help='Data')
    parser.add_argument('--neutral_words', type=str)
    parser.add_argument('--attribute_words', type=tp, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta', 'gpt2'])
    parser.add_argument('--ab_test_type', type=str, default='final',
                        choices=['raw', 'reliability', 'quality', 'quantity-100', 'quantity-1000', 'quantity-10000', 'final'])

    args = parser.parse_args()

    return args

def prepare_tokenizer(args):
    if args.model_type == 'bert':
        pretrained_weights = 'bert-large-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'roberta':
        pretrained_weights = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'gpt2':
        pretrained_weights = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_weights)
    return tokenizer

def main(args):
    SUPERVISED_ENTITIES = []
    supervised_entities = [w.lower() for w in SUPERVISED_ENTITIES]
    entity_count = {}

    data = [l.strip() for l in open(args.input)]
    if args.neutral_words:
        neutrals = [word.strip() for word in open(args.neutral_words)]
        neutral_set = set(neutrals)

    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    sequential_l = []
    attributes_l = []
    all_attributes_set = set()
    for attribute in args.attribute_words:
        l = [word.strip() for word in open(attribute)]
        sequential_l.append(l)
        attributes_l.append(set(l))
        all_attributes_set |= set(l)

    valid_lines = []

    for orig_line in tqdm(data, desc='sentence'):
        # neutral_flag = True
        orig_line = orig_line.strip()
        if len(orig_line) < 1:
            continue
        leng = len(orig_line.split())
        if leng > args.block_size or leng <= 1:
            continue
        tokens_orig = [token.strip() for token in re.findall(pat, orig_line)]
        tokens_lower = [token.lower() for token in tokens_orig]
        token_set = set(tokens_lower)

        for i, attribute_set in enumerate(attributes_l):
            if attribute_set & token_set and neutral_set & token_set:
                valid_lines.append(orig_line)

    with open(args.output+'/data.txt', 'w') as f:
        for line in list(set(valid_lines)):
            f.write(line)
            f.write('\n')
if __name__ == "__main__":
    args = parse_args()
    main(args)

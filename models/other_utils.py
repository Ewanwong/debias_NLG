from sklearn.model_selection import train_test_split
import regex as re

def load_file_to_list(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        outputs = f.read().strip().split('\n')
    return outputs

def get_intervals(sent, toks):
    prefix_length = len(toks[0])
    prefix = toks[0]
    intervals = [0] # record where to put whitespace
    if len(toks) == 1:
        return intervals
    for i in range(1, len(toks)-1):
        if sent[prefix_length] == ' ':
            intervals.append(1)
            prefix = prefix + ' ' + toks[i]
            prefix_length += len(toks[i]) + 1
        else:
            intervals.append(0)
            prefix += toks[i]
            prefix_length += len(toks[i])
        
        """
        if prefix_length+len(toks[i])+1 > len(sent):
            intervals.append(0)
            prefix += toks[i]
            prefix_length += len(toks[i])
        elif prefix + ' ' + toks[i] == sent[:prefix_length+len(toks[i])+1]:
            intervals.append(1)
            prefix = prefix + ' ' + toks[i]
            prefix_length += len(toks[i]) + 1
        else:
            intervals.append(0)
            prefix += toks[i]
            prefix_length += len(toks[i])
        """
    intervals.append(0)
    return intervals

def train_dev_split(data_path, gender_swapped_data_path, ratio=0.9, shuffle=True):
    data = load_file_to_list(data_path)
    gender_swapped_data = load_file_to_list(gender_swapped_data_path)
    data, gender_swapped_data = clean_data(data, gender_swapped_data)
    assert len(data) == len(gender_swapped_data)
    num =  len(data)
    index_list = list(range(num))
    train_index, dev_index = train_test_split(index_list, train_size=ratio, shuffle=shuffle)

    train_data, train_gender_swapped_data = [data[i] for i in train_index], [gender_swapped_data[i] for i in train_index]
    dev_data, dev_gender_swapped_data = [data[i] for i in dev_index], [gender_swapped_data[i] for i in dev_index]

    return train_data, train_gender_swapped_data, dev_data, dev_gender_swapped_data

def clean_data(data, gs_data):
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    new_data = []
    new_gender_swapped_data = []

    for s1, s2 in zip(data, gs_data):
        s1_tokens, s2_tokens = [tok.strip().lower() for tok in re.findall(pat, s1)], [tok.strip().lower() for tok in re.findall(pat, s2)]
        if len(s1_tokens) != len(s2_tokens):
            continue
        else: 
            new_data.append(s1)
            new_gender_swapped_data.append(s2)

    return new_data, new_gender_swapped_data
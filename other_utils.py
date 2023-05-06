def load_file_to_list(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        outputs = f.read().strip().split('\n')
    return outputs

def get_intervals(sent, toks):
    prefix_length = len(toks[0])
    prefix = toks[0]
    intervals = [0] # record where to put whitespace
    for i in range(1, len(toks)):
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
    return intervals
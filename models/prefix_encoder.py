import torch
import os

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
    def save(self, save_path):
        save_bin_path = os.path.join(save_path, 'pytorch_model.bin')
        if self.prefix_projection:
            torch.save({"embedding_matrix":self.embedding.state_dict(), "trans_matrix":self.trans.state_dict()}, save_bin_path)
        else:
            torch.save({"embedding_matrix":self.embedding.state_dict()}, save_bin_path)
        
        
    
    def load(self, save_path):
        save_bin_path = os.path.join(save_path, 'pytorch_model.bin')
        if self.prefix_projection:
            checkpoint = torch.load(save_bin_path)
            self.embedding.load_state_dict(checkpoint["embedding_matrix"])
            self.trans.load_state_dict(checkpoint['trans_matrix'])
            
        else:
            checkpoint = torch.load(save_bin_path)
            self.embedding.load_state_dict(checkpoint["embedding_matrix"])
    
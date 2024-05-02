import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Optional, Tuple, Union
from transformers import AutoConfig
import os
import json
# from transformers import AutoModelWithLMHead, PreTrainedModel

from models.prefix_encoder import PrefixEncoder

class PrefixGPT2(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

        self.pre_seq_len = config.pre_seq_len
        self.n_layer= config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.dropout = torch.nn.Dropout(config.resid_pdrop)
        self.prefix_encoder = PrefixEncoder(config)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        for param in self.gpt2.parameters():
            param.requires_grad = False

        # self.init_weights()
        

    def get_prefix(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.gpt2.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        
        return past_key_values
    
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        batch_size = input_ids.shape[0]

        if past_key_values is not None:
            raise ValueError("Past key values are for other use")
        else:
            past_key_values = self.get_prefix(batch_size=batch_size)
        #prefix_attention_mask = torch.ones(batch_size, past_key_values[0].shape[-2]).to(self.gpt2.device) # self.pre_seq_len
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.gpt2.device)
        
        if attention_mask is not None:
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        """
        if attention_mask is None:
            attention_mask = torch.cat((prefix_attention_mask, torch.ones(batch_size, input_ids.shape[1]).to(self.gpt2.device)), dim=1)
        elif attention_mask.shape[1] != (self.pre_seq_len + input_ids.shape[1]):
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        """
        outputs = self.gpt2(input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            )
        
        return outputs
    
    def save_pretrained(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.prefix_encoder.save(save_path)
        self.config.save_pretrained(save_path)
    
    def load(self, save_path):
        self.prefix_encoder.load(save_path)

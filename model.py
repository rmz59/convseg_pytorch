import logging
import torch
import torch.nn as nn

from layers import ConvGLUBlock
from crf import CRF


class CharWordSeg(nn.Module):
    def __init__(self, vocab_tag, char_embed_size, num_hidden_layer, channel_size, kernel_size, dropout_rate=0.2):
        super(CharWordSeg, self).__init__()
        self.vocab_tag = vocab_tag
        num_tags = len(self.vocab_tag['tag_to_index'])
        vocab_size = len(self.vocab_tag['token_to_index'])
        self.char_embedding = nn.Embedding(vocab_size, char_embed_size)
        self.num_hidden_layer = num_hidden_layer
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.dropout_embed = nn.Dropout(dropout_rate)
        self.glu_layers = nn.ModuleList([ConvGLUBlock(in_channels=char_embed_size, 
                                                    out_channels=channel_size, 
                                                    kernel_size=kernel_size, 
                                                    drop_out=0.2,
                                                    padding=1)]
                                        + [ConvGLUBlock(in_channels=channel_size, 
                                                     out_channels=channel_size, 
                                                     kernel_size=kernel_size,
                                                     drop_out=0.2,
                                                     padding=1) for _ in range(num_hidden_layer-1)])
        self.hidden_to_tag = nn.Linear(char_embed_size, num_tags)
        self.crf_layer =  CRF(num_tags, batch_first=True)

    def forward(self, input_sentences, tags, mask):
        """ 
        Args:
            input_sentences: torch.tensor -> shape: (batch_size, max_sent_length)
            tags: torch.tensor -> (batch_size, max_sent_length)
            masks: torch.tensor -> (batch_size, max_sent_length)

        Example:
        >>> vocab_size = 10
        >>> char_embed_size = 5
        >>> tag_size = 3
        >>> batch_size = 3
        >>> max_sent_size = 5
        >>> input_sents = torch.tensor([[3, 9, 8, 3, 4],[9, 2, 7, 0, 1],[9, 7, 9, 1, 6]])
        >>> model = CharWordSeg(vocab_size, char_embed_size, 2, 5, 3)
        >>> model(input_sents).shape
        torch.Size([3, 5, 3])
        """
        x = self.dropout_embed(self.char_embedding(input_sentences))  # X shape: (batch_size, max_sent_length, char_embed_size)

        # glu layers
        for glu_layers in self.glu_layers:
            x = glu_layers(x)
        # output dim: (batch_size, max_sent_length, channel_size)

        emission = self.hidden_to_tag(x)   # output dim: (batch_size, max_sen_length, num_tags)
        x = self.crf_layer(emission, tags, mask)
        return x
    
    def decode(self, input_sentences, mask):
        x = self.dropout_embed(self.char_embedding(input_sentences))  # X shape: (batch_size, max_sent_length, char_embed_size)

        # glu layers
        for glu_layers in self.glu_layers:
            x = glu_layers(x)
        # output dim: (batch_size, max_sent_length, channel_size)

        emission = self.hidden_to_tag(x)   # output dim: (batch_size, max_sen_length, num_tags)
        x = self.crf_layer.decode(emission, mask) # output dim: (batch_size, max_sen_length)
        return x        

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CharWordSeg(vocab_tag=params['vocab_tag'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        logging.info(f'save model parameters to [{path}]')

        params = {
            'args': dict(char_embed_size=self.char_embedding, 
                         num_hidden_layer=self.num_hidden_layer, 
                         channel_size = self.channel_size,
                         kernel_size = self.kernel_size,
                         dropout_rate=self.dropout_rate),
            'vocab_tag': self.vocab_tag,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

import torch.nn as nn

from layers import ConvGLUBlock


class CharWordSeg(nn.Module):
    def __init__(self, vocab_size, char_embed_size, num_hidden_layer, channel_size, kernel_size, drop_out=0.2, num_tags=3):
        super(CharWordSeg, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embed_size)
        self.dropout_embed = nn.Dropout(drop_out)
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

    def forward(self, input_sentences):
        """ 
        Args:
            input_sentences: List[List(int)] -> shape: (batch_size, max_sent_length)

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

        x = self.hidden_to_tag(x)   # output dim: (batch_size, max_sen_length, num_tags)
        return x
        
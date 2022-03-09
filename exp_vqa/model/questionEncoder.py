import torch
import torch.nn as nn
import numpy as np
from itertools import chain
from utils.misc import reverse_padded_sequence


class BiGRUEncoder(nn.Module):
    """
    遵循Explainable Neural Computation via Stack Neural Module Networks模型架构
    使用BiLSTM将问题q转换为一个d维序列（这里使用双向GRU）
    [h_1, ..., h_S] = BiLSTM(q; theta_BiLSTM)
    """
    def __init__(self, dim_word, dim_hidden):
        super().__init__()
        self.forward_gru = nn.GRU(dim_word, dim_hidden // 2)
        self.backward_gru = nn.GRU(dim_word, dim_hidden // 2)
        for name, param in chain(self.forward_gru.named_parameters(), self.backward_gru.named_parameters()):
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_seqs, input_embedded, input_seq_lens):
        """
            Input:
                input_seqs: [seq_max_len, batch_size]
                input_seq_lens: [batch_size]
        """
        embedded = input_embedded  # [seq_max_len, batch_size, word_dim]
        # 只需要取前向GRU最后的输出，不需要隐藏层了
        forward_outputs = self.forward_gru(embedded)[0]  # [seq_max_len, batch_size, dim_hidden/2]
        backward_embedded = reverse_padded_sequence(embedded, input_seq_lens)
        backward_outputs = self.backward_gru(backward_embedded)[0]
        backward_outputs = reverse_padded_sequence(backward_outputs, input_seq_lens)
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)  # [seq_max_len, batch_size, dim_hidden]
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(input_seq_lens):
            hidden.append(
                torch.cat([forward_outputs[l - 1, i], backward_outputs[0, i]], dim=0)
            )
        hidden = torch.stack(hidden)  # (batch_size, dim)
        return outputs, hidden

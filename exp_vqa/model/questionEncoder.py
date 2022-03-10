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
        # forward_outputs的维度是（seq_max_len, batch_size, dim_hidden/2）
        forward_outputs = self.forward_gru(embedded)[0]  # [seq_max_len, batch_size, dim_hidden/2]

        # 将输入的embedding逆序得到backward_embedded
        backward_embedded = reverse_padded_sequence(embedded, input_seq_lens)

        # 再将backward_embedded通过backward_gru，取输出
        backward_outputs = self.backward_gru(backward_embedded)[0]

        # 将backward_embedded通过backward_gru得到的结果逆序
        backward_outputs = reverse_padded_sequence(backward_outputs, input_seq_lens)

        # 然后拼接前向GRU的结果和后向GRU的结果
        # 这时输出的维度就是[seq_max_len, batch_size, dim_hidden]
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)  # [seq_max_len, batch_size, dim_hidden]
        # indexing outputs via input_seq_lens

        hidden = []
        for i, l in enumerate(input_seq_lens):
            # 取一个batch中每一个embedding的序列长度为l
            # i从0到batch_size - 1
            hidden.append(
                # forward_outputs[l - 1, i]的维度是dim_hidden/2
                # backward_outputs[0, i]的维度也是dim_hidden/2
                # 第0维拼接就是（2，dim_hidden/2）
                torch.cat([forward_outputs[l - 1, i], backward_outputs[0, i]], dim=0)
            )
        # 堆叠起来的hidden就是（batch_size, 2, dim_hidden/2）
        hidden = torch.stack(hidden)  # (batch_size, dim)
        return outputs, hidden

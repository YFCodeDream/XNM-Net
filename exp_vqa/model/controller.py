import numpy as np
import torch
from torch import nn
from itertools import chain


class Controller(nn.Module):
    """
    遵循Explainable Neural Computation via Stack Neural Module Networks模型架构
    Sec3.1 布局控制器
    从net.py传来的参数
    controller_kwargs = {
            'num_module': len(self.module_names), 值为6
            'dim_lstm': self.dim_hidden, 值为1024
            'T_ctrl': self.T_ctrl, 默认为3，controller decode length
                （对应原论文里的时间步time-step）
            'use_gumbel': self.use_gumbel, 默认为False，whether use gumbel softmax for module prob
        }
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.num_module = kwargs['num_module']
        self.dim_lstm = kwargs['dim_lstm']
        self.T_ctrl = kwargs['T_ctrl']
        self.use_gumbel = kwargs['use_gumbel']

        self.fc_q_list = []  # W_1^{(t)} q + b_1

        for t in range(self.T_ctrl):
            # 依据控制器解码的长度（time-step），添加对应数量的线性层，线性层的输入输出维度都是1024
            # 每一个时间步的权重W_1^(t)彼此不同
            self.fc_q_list.append(nn.Linear(self.dim_lstm, self.dim_lstm))
            # 将线性层注册成网络模块
            self.add_module('fc_q_%d' % t, self.fc_q_list[t])

        # 这里对应原文的：
        # 以循环方式从时间步t=0到时间步t=T-1
        # 在每一个时间步t，对问题q应用与时间步相关的线性变换（存储在fc_q_list里的线性层）
        # 并将其与先前的d维文本参数c_(t-1)线性组合 u = W_2[W_1^(t)q + b_1; c_(t-1)] + b_2
        # 所以这里的W_2的维度是d*2d
        self.fc_q_cat_c = nn.Linear(2 * self.dim_lstm, self.dim_lstm)  # W_2 [q;c] + b_2

        # 选择在当前时间步执行的模块，将MLP应用于之前的u
        # 生成num_module维的预测向量ω^(t)
        self.fc_module_weight = nn.Sequential(
            nn.Linear(self.dim_lstm, self.dim_lstm),
            nn.ReLU(),
            nn.Linear(self.dim_lstm, self.num_module)
        )

        # 共S个单词，计算第s个单词对应的cv_(t,s)，即W_3
        self.fc_raw_cv = nn.Linear(self.dim_lstm, 1)

        # 文本参数的初始化
        self.c_init = nn.Parameter(torch.zeros(1, self.dim_lstm).normal_(mean=0, std=np.sqrt(1 / self.dim_lstm)))

    def forward(self, lstm_seq, q_encoding, embed_seq, seq_length_batch):
        """        
        Input:
            lstm_seq: [seq_max_len, batch_size, d]
            q_encoding: [batch_size, d]
            embed_seq: [seq_max_len, batch_size, e]
            seq_length_batch: [batch_size]
        """
        device = lstm_seq.device
        # 这里使用的是batch_first为False的数据，所以lstm_seq.size(1)才是batch_size
        batch_size, seq_max_len = lstm_seq.size(1), lstm_seq.size(0)

        seq_length_batch = seq_length_batch.view(1, batch_size).expand(seq_max_len,
                                                                       batch_size)  # [seq_max_len, batch_size]

        # 扩展到batch_size个初始化文本参数
        c_prev = self.c_init.expand(batch_size, self.dim_lstm)  # (batch_size, dim)

        # 初始化各项存储列表
        module_logit_list = []
        module_prob_list = []
        c_list, cv_list = [], []

        for t in range(self.T_ctrl):
            # 将question_embedding按照时间步顺序过线性层
            q_i = self.fc_q_list[t](q_encoding)

            # 拼接过完线性层W_1^(t)之后的结果和文本参数
            # [W_1^(t)q + b_1; c_(t-1)]
            q_i_c = torch.cat([q_i, c_prev], dim=1)  # [batch_size, 2d]

            # 接着过第二个线性层，，获得结果u，输出维度是d
            cq_i = self.fc_q_cat_c(q_i_c)  # [batch_size, d]

            # 获取预测向量，包含各个模块的权重分布，存储在module_prob里
            # module_logit是经过softmax前的logits向量
            module_logit = self.fc_module_weight(cq_i)  # [batch_size, num_module]
            module_prob = nn.functional.gumbel_softmax(module_logit, hard=self.use_gumbel)  # [batch_size, num_module]

            # 这里是u与h_s的哈达玛积
            elem_prod = cq_i.unsqueeze(0) * lstm_seq  # [seq_max_len, batch_size, dim]

            # 输入的问题q有S个单词，每一个单词都算一个双向GRU编码，得到S个cv_(t,s)，所以raw_cv_i的每一列都是每一个question的每一个单词对应的logits
            raw_cv_i = self.fc_raw_cv(elem_prod).squeeze(2)  # [seq_max_len, batch_size]

            # 计算有效单词数矩阵，因为每个question的编码序列长短不一，之前padding成最长的问题编码序列长度
            # 所以在计算之后的softmax分数时，只能在每个question有的单词之间计算，不能把后面的padding也算上
            # 所以无效的padding单元在之后需要设置为-inf，过了softmax后权重就变为0，不会被选择到
            invalid_mask = torch.arange(seq_max_len).long().to(device).view(-1, 1).expand_as(raw_cv_i).ge(
                seq_length_batch)

            # 将无效的padding置为-inf，过了softmax就归零了
            raw_cv_i.data.masked_fill_(invalid_mask, -float('inf'))

            # 对每一列计算softmax，每一列就成为每一个问题中所有单词对应的权重
            cv_i = nn.functional.softmax(raw_cv_i, dim=0).unsqueeze(2)  # [seq_max_len, batch_size, 1]

            # c_t = sigma_{s=1}^S(cv_(t,s) * h_s))
            c_i = torch.sum(lstm_seq * cv_i, dim=0)  # [batch_size, d]

            # c_i的维度是（batch_size, dim_lstm）
            assert c_i.size(0) == batch_size and c_i.size(1) == self.dim_lstm

            # 更新文本参数
            c_prev = c_i

            # add into results 将每个时间步的结果存入列表
            module_logit_list.append(module_logit)
            module_prob_list.append(module_prob)
            c_list.append(c_i)
            # cv_list每一个元素的维度是（batch_size, seq_max_len），每一行是单词的权重
            cv_list.append(cv_i.squeeze(2).permute(1, 0))

        return (torch.stack(module_logit_list),  # [T_ctrl, batch_size, num_module]
                torch.stack(module_prob_list),  # [T_ctrl, batch_size, num_module]
                torch.stack(c_list),  # [T_ctrl, batch_size, d]
                torch.stack(cv_list))  # [T_ctrl, batch_size, seq_max_len]

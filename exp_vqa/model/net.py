import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from . import composite_modules as modules
from .questionEncoder import BiGRUEncoder
from .controller import Controller


# noinspection PyIncorrectDocstring,PyProtectedMember
class XNMNet(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_v, # vertex and edge embedding of scene graph 默认是512
             dim_word, # word embedding 默认300
             dim_hidden, # hidden of seq2seq 默认1024
             dim_vision, 默认2048
             dim_edge, 默认256
             glimpses, 默认2
             cls_fc_dim, 融合后中间层的维数 默认1024
             dropout_prob, 默认0.5
             T_ctrl, 默认3
             stack_len, 默认为4
             device,
             use_gumbel, 默认False
             use_validity, 默认True
        """
        super().__init__()
        for k, v in kwargs.items():
            # 将传入的参数初始化为属性
            setattr(self, k, v)

        # vocab['answer_token_to_idx']，出现过的answer个数（默认为3000）
        self.num_classes = len(self.vocab['answer_token_to_idx'])

        # 在最后答案预测时使用
        self.classifier = Classifier(
            in_features=(self.glimpses * self.dim_vision, self.dim_hidden),
            mid_features=self.cls_fc_dim,
            out_features=self.num_classes,
            drop=self.dropout_prob
        )

        # 视觉特征到点的表示的映射
        self.map_vision_to_v = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.dim_vision, self.dim_v, bias=False),
        )
        # 两个点的表示到一条边的表示的映射
        self.map_two_v_to_edge = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            # 这里对应两个node embedding的拼接维度
            nn.Linear(self.dim_v * 2, self.dim_edge, bias=False),
        )
        # 取出所有在question中出现的token的数量
        self.num_token = len(self.vocab['question_token_to_idx'])
        # token embedding在train.py中将权重设置为glove的预训练权重
        # dim_word应该是300维，对应glove的embedding维度
        self.token_embedding = nn.Embedding(self.num_token, self.dim_word)
        self.dropout = nn.Dropout(self.dropout_prob)

        # modules
        # Sec.3中提到的各种模块
        # （这一部分核心模块，有点难看）-------------------------------------------------------------------------------------
        # 取输入的模块种类数
        # MODULE_INPUT_NUM = {
        #     '_NoOp': 1,
        #     '_Find': 0,
        #     '_Transform': 1,
        #     '_Filter': 1,
        #     '_And': 2,
        #     '_Describe': 1,
        # }
        self.module_names = modules.MODULE_INPUT_NUM.keys()
        self.num_module = len(self.module_names)

        # module_funcs获取在composite_modules里MODULE_INPUT_NUM键值中的所有模块的传参实例化对象，参数由kwargs指定
        self.module_funcs = [getattr(modules, m[1:] + 'Module')(**kwargs) for m in self.module_names]
        # stack_len默认为4
        self.module_validity_mat = modules._build_module_validity_mat(self.stack_len, self.module_names)
        self.module_validity_mat = torch.Tensor(self.module_validity_mat).to(self.device)

        for name, func in zip(self.module_names, self.module_funcs):
            # e.g. 相当于self.name = func
            self.add_module(name, func)

        # question encoder dim_word为300，dim_hidden为1024
        self.question_encoder = BiGRUEncoder(self.dim_word, self.dim_hidden)

        # controller
        controller_kwargs = {
            'num_module': len(self.module_names),
            'dim_lstm': self.dim_hidden,
            'T_ctrl': self.T_ctrl,
            'use_gumbel': self.use_gumbel,
        }
        self.controller = Controller(**controller_kwargs)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1 / np.sqrt(self.dim_word))

    def forward(self, questions, questions_len, vision_feat, relation_mask):
        """
        Args:
            questions [Tensor] (batch_size, seq_len)
            questions_len [Tensor] (batch_size)
            vision_feat (batch_size, dim_vision, num_feat)
            relation_mask (batch_size, num_feat, num_feat)
        """
        batch_size = len(questions)
        questions = questions.permute(1, 0)  # (seq_len, batch_size)
        questions_embedding = self.token_embedding(questions)  # (seq_len, batch_size, dim_word)
        questions_embedding = torch.tanh(self.dropout(questions_embedding))
        questions_outputs, questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            questions_outputs, questions_hidden, questions_embedding, questions_len)

        # feature processing
        vision_feat = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        feat_inputs = vision_feat.permute(0, 2, 1)
        if self.dim_v != self.dim_vision:
            feat_inputs = self.map_vision_to_v(feat_inputs)  # (batch_size, num_feat, dim_v)
        num_feat = feat_inputs.size(1)
        feat_inputs_expand_0 = feat_inputs.unsqueeze(1).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_inputs_expand_1 = feat_inputs.unsqueeze(2).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3)  # (bs, num_feat, num_feat, 2*dim_v)
        feat_edge = self.map_two_v_to_edge(feat_edge)

        # stack initialization
        att_stack = torch.zeros(batch_size, num_feat, self.glimpses, self.stack_len).to(self.device)
        stack_ptr = torch.zeros(batch_size, self.stack_len).to(self.device)
        stack_ptr[:, 0] = 1
        mem = torch.zeros(batch_size, self.glimpses * self.dim_vision).to(self.device)

        # cache for visualization
        cache_module_prob = []
        cache_att = []

        for t in range(self.T_ctrl):
            c_i = c_list[t]  # (batch_size, dim_hidden)
            module_logit = module_logits[t]  # (batch_size, num_module)
            if self.use_validity:
                if t < self.T_ctrl - 1:
                    module_validity = torch.matmul(stack_ptr, self.module_validity_mat)
                    module_validity[:, 5] = 0
                else:  # last step must describe
                    module_validity = torch.zeros(batch_size, self.num_module).to(self.device)
                    module_validity[:, 5] = 1
                module_invalidity = (1 - torch.round(module_validity)).byte()  # hard validate
                module_logit.masked_fill_(module_invalidity, -float('inf'))
                module_prob = F.gumbel_softmax(module_logit, hard=self.use_gumbel)
            else:
                module_prob = module_probs[t]
            module_prob = module_prob.permute(1, 0)  # (num_module, batch_size)

            # run all modules
            res = [
                f(vision_feat.permute(0, 2, 1), feat_inputs, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem)
                for f in self.module_funcs]
            att_stack_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1, 1, 1) * torch.stack([r[0] for r in res]), dim=0)
            stack_ptr_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[1] for r in res]), dim=0)
            stack_ptr_avg = modules._sharpen_ptr(stack_ptr_avg, hard=False)
            mem_avg = torch.sum(module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[2] for r in res]),
                                dim=0)
            att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg
            # cache for visualization
            cache_module_prob.append(module_prob)
            atts = []
            for r in res:
                att = modules._read_from_stack(r[0], r[1])  # (batch_size, att_dim, glimpse)
                atts.append(att)
            cache_att.append(atts)

        # Part 1. features from scene graph module network. (batch, dim_v)
        # Part 2. question prior. (batch, dim_hidden)
        predicted_logits = self.classifier(mem, questions_hidden)
        others = {
            'module_prob': cache_module_prob,  # (T, num_module, batch_size)
            'att': cache_att,  # (T, num_module, batch_size, att_dim, glimpse)
            'cv': cv_list,  # (T, batch_size, len)
        }
        return predicted_logits, others


"""
The Classifier and Attention is according to the codes of Learning to Count 
[https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/model.py]
"""


# noinspection PyMethodMayBeStatic
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x与y同维度，进行如下运算
        return - (x - y) ** 2 + F.relu(x + y)


# noinspection PyMethodOverriding
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop):
        # in_features是个元组，存储两个部分的feature维度
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, x, y):
        # self.lin11(self.drop(x))和self.lin12(self.drop(y))都是以mid_features为输出维度
        x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        # 过一个bn层，一个dropout，最后输出维度为out_features
        x = self.lin2(self.drop(self.bn(x)))
        return x

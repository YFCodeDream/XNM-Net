import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from . import composite_modules as modules
from .questionEncoder import BiGRUEncoder
from .controller import Controller


# noinspection PyIncorrectDocstring,PyProtectedMember,GrazieInspection
class XNMNet(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_v, # vertex and edge embedding of scene graph 默认是512
             dim_word, # word embedding 默认300
             dim_hidden, # hidden of seq2seq 默认1024
             dim_vision, 默认2048，spatial为True，则为2053
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
        # 先把问题编码序列的每一个编码转成one hot向量（维度为num_token），再乘上embedding矩阵，得到分布式表示
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
        # stack_len默认为4，module_validity_mat维度为（4，6）
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

        # 模块参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1 / np.sqrt(self.dim_word))

    def forward(self, questions, questions_len, vision_feat, relation_mask):
        """
        Args:
            questions [Tensor] (batch_size, seq_len) seq_len：最大的问题编码序列长度
            questions_len [Tensor] (batch_size)
            vision_feat (batch_size, dim_vision, num_feat) (batch_size, 2053(2048+5), 36)
            relation_mask (batch_size, num_feat, num_feat) (batch_size, 36, 36)

            # question：转成np.array格式的填充完毕的question每一个token编码转成的列表，填充长度是最大的问题编码序列长度
            # questions_len：转换为np.array格式的每一个question的长度的列表
            # vision_feat：如果使用空间特征，就把对应的x1,y1,x2,y2,bounding box面积拼接在2048维特征后，变成2053维
            # relation_mask：36个bounding box是否有重叠的标记矩阵，有重叠则在对应位置记为1，为对称矩阵
        """
        batch_size = len(questions)

        # permute是将0维和1维换顺序
        questions = questions.permute(1, 0)  # (seq_len, batch_size)

        questions_embedding = self.token_embedding(questions)  # (seq_len, batch_size, dim_word)

        questions_embedding = torch.tanh(self.dropout(questions_embedding))

        # 这里questions_embedding的size是(seq_len, batch_size, dim_word)
        questions_outputs, questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)

        # 从布局控制器里获取存有所有时间步的运行结果
        module_logits, module_probs, c_list, cv_list = self.controller(
            questions_outputs, questions_hidden, questions_embedding, questions_len)

        # feature processing 特征处理
        # 此时的vision_feat是(batch_size, 2053, 36)维度
        # vision_feat.norm(p=2, dim=1, keepdim=True)对第1维，即每个bounding box对应的2053维特征求平方和再开根号
        # 起到归一化的作用，1e-12防止被除数为0
        vision_feat = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)

        # feat_inputs: (batch_size, 36, 2053)
        feat_inputs = vision_feat.permute(0, 2, 1)

        # dim_v一开始默认为512，dim_vision为2053（spatial为True）
        if self.dim_v != self.dim_vision:
            # Dropout+Linear，过一个线性层，将2053维转成512维
            # feat_inputs维度为（batch_size, 36, 512）
            feat_inputs = self.map_vision_to_v(feat_inputs)  # (batch_size, num_feat, dim_v)

        # num_feat = 36
        num_feat = feat_inputs.size(1)

        # feat_inputs_expand_0将一个batch里的每一个(36, 512)的特征拷贝了num_feat份，所以一个batch里每个元素都是(36, 36, 512)维
        # feat_inputs_expand_0最后两维(36, 512)仍然对应着feat_inputs的最后两维(36, 512)
        feat_inputs_expand_0 = feat_inputs.unsqueeze(1).expand(batch_size, num_feat, num_feat, self.dim_v)

        # feat_inputs_expand_1将一个batch里的图像特征中的每一个bounding box的特征拷贝了num_feat份
        # feat_inputs_expand_1最后两维(36, 512)仅仅是将一个bounding box的512维特征拷贝了36份，每一行都相同，第1维的36对应着feat_inputs的第1维
        feat_inputs_expand_1 = feat_inputs.unsqueeze(2).expand(batch_size, num_feat, num_feat, self.dim_v)

        # 这里对应Explainable and Explicit Visual Reasoning over Scene Graphs的Sec3.1
        # 将DET场景图的边表示为e_{i, j}=[v_i; v_j]，仅连接两个点的特征向量作为edge embedding
        # 其实依据代码来看，并不能称之为严格的场景图，应将其称之为一张图像的bounding box的相关图
        feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3)  # (bs, num_feat, num_feat, 2*dim_v)

        # Dropout+Linear，过一个线性层，将2*512维转成256维
        # feat_edge维度为(batch_size, 36, 36, 256)
        feat_edge = self.map_two_v_to_edge(feat_edge)

        # stack initialization 初始化可微堆栈，glimpses默认为2
        # att_stack维度: (batch_size, 36, 2, 4)，初始化为全零
        att_stack = torch.zeros(batch_size, num_feat, self.glimpses, self.stack_len).to(self.device)

        # stack_ptr存batch_size个栈顶指针p，每个p为stack_len维的one hot向量，维度为(batch_size, stack_len(4))
        stack_ptr = torch.zeros(batch_size, self.stack_len).to(self.device)

        # 初始化栈顶指针p在第一个元素
        # 对应Explainable Neural Computation via Stack Neural Module Networks的Sec3.3
        stack_ptr[:, 0] = 1

        # mem维度(batch_size, 2*2053)，初始化为全零张量
        mem = torch.zeros(batch_size, self.glimpses * self.dim_vision).to(self.device)

        # cache for visualization 用于可视化的缓存
        cache_module_prob = []
        cache_att = []

        # 遍历所有时间步
        for t in range(self.T_ctrl):
            # 取出第t个时间步的文本参数，维度为(batch_size, dim_lstm(即dim_hidden))
            c_i = c_list[t]  # (batch_size, dim_hidden)

            # 取出第t个时间步的经过softmax前的logits向量，表示所有模块的权重
            module_logit = module_logits[t]  # (batch_size, num_module)

            # use_validity默认为True
            if self.use_validity:
                if t < self.T_ctrl - 1: # 如果当前时间步不是最后一个时间步
                    # (batch_size, stack_len(4)) * (stack_len(4), num_module(6))
                    # 计算得到module_validity，为模块的有效布尔向量，只有在模块有效的范围内才能对元素进行操作
                    # module_validity用以标记当前batch中每一个样本（一行对应一个样本），哪一个模块是有效的
                    module_validity = torch.matmul(stack_ptr, self.module_validity_mat)

                    # 将最后一列的模块（Describe）置为无效
                    module_validity[:, 5] = 0
                else:  # last step must describe
                    # 最后一步必须为Describe，因此直接将其余模块有效性置零，将最后一列的模块（Describe）有效性置一
                    module_validity = torch.zeros(batch_size, self.num_module).to(self.device)
                    module_validity[:, 5] = 1

                # 将模块有效性取反再赋值给module_invalidity
                module_invalidity = (1 - torch.round(module_validity)).byte()  # hard validate

                # 将无效的模块的logit向量置为-inf，过一个softmax就归零了
                module_logit.masked_fill_(module_invalidity, -float('inf'))

                module_prob = F.gumbel_softmax(module_logit, hard=self.use_gumbel)
            else:
                # 如果不用有效性检验，就直接取布局控制器的输出
                module_prob = module_probs[t]

            # 原来的module_prob是(batch_size, num_module)
            module_prob = module_prob.permute(1, 0)  # (num_module, batch_size)

            # run all modules 执行所有模块

            # vision_feat.permute(0, 2, 1)维度是(batch_size, 36, 2053)
            # feat_inputs维度为（batch_size, 36, 512）
            # feat_edge维度为(batch_size, 36, 36, 256)
            # c_i: 第t个时间步的文本参数，维度为(batch_size, dim_lstm(即dim_hidden))
            # relation_mask维度为(batch_size, 36, 36)
            # att_stack维度: (batch_size, 36, 2, 4)
            # stack_ptr存batch_size个栈顶指针p，每个p为stack_len维的one hot向量，维度为(batch_size, stack_len(4))
            # mem维度(batch_size, 2*2053)

            # 在每个模块上执行运算，获得对应的结果
            # 返回注意力堆栈att_stack, 堆栈指针stack_ptr, mem
            res = [
                f(vision_feat.permute(0, 2, 1), feat_inputs, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem)
                for f in self.module_funcs]

            # 将每一个堆栈的结果乘上对应的权重，做加权求和
            att_stack_avg = torch.sum(
                # r[0]为att_stack
                module_prob.view(self.num_module, batch_size, 1, 1, 1) * torch.stack([r[0] for r in res]), dim=0)

            # 将堆栈指针也进行加权求和
            stack_ptr_avg = torch.sum(
                # r[1]为stack_ptr堆栈指针
                module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[1] for r in res]), dim=0)

            stack_ptr_avg = modules._sharpen_ptr(stack_ptr_avg, hard=False)

            # 其实就是Describe模块的结果乘上Describe模块的权重
            mem_avg = torch.sum(module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[2] for r in res]),
                                dim=0)

            # 更新中间结果(att_stack, stack_ptr, mem)
            att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg

            # cache for visualization
            # 保存每次的模块权重
            cache_module_prob.append(module_prob)
            atts = []
            for r in res:
                # 取出当前栈指针指向的注意
                att = modules._read_from_stack(r[0], r[1])  # (batch_size, att_dim, glimpse)
                atts.append(att)
            cache_att.append(atts)

        # Part 1. features from scene graph module network. (batch, dim_v)
        # Part 2. question prior. (batch, dim_hidden)
        # 运行完指定时间步之后，融合bounding box相关图特征以及问题特征
        # 预测对应答案
        predicted_logits = self.classifier(mem, questions_hidden)

        # 将其余的推理过程中的数据返回
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

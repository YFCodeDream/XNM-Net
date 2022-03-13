import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.misc import convert_to_one_hot
from IPython import embed

# MODULE_INPUT_NUM对应这个模块需要的输入的元素个数
MODULE_INPUT_NUM = {
    '_NoOp': 1,
    '_Find': 0,
    '_Transform': 1,
    '_Filter': 1,
    '_And': 2,
    '_Describe': 1,
}

# MODULE_OUTPUT_NUM对应这个模块的输出元素个数
MODULE_OUTPUT_NUM = {
    '_NoOp': 0,
    '_Find': 1,
    '_Transform': 1,
    '_Filter': 1,
    '_And': 1,
    '_Describe': 1,
}

"""
att_stack: (batch_size, dim_att, glimpse, stack_len)
att: (batch_size, dim_att, glimpse)
stack_ptr: (batch_size, stack_len)
mem: (batch_size, dim_vision * glimpse)
"""


# 所有的模块中的forward函数都是以下形式
# def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):

# vision_feat经过permute(0, 2, 1)，维度是(batch_size, 36, 2053)
# feat_inputs维度为（batch_size, 36, 512）
# feat_edge维度为(batch_size, 36, 36, 256)
# c_i: 第t个时间步的文本参数，维度为(batch_size, dim_lstm(即dim_hidden))
# relation_mask维度为(batch_size, 36, 36)

# att_stack维度: (batch_size, 36(num_feat), 2(glimpse), 4(stack_len))
# stack_ptr存batch_size个栈顶指针p，每个p为stack_len维的one hot向量，维度为(batch_size, stack_len(4))
# mem维度(batch_size, 2*2053)

# 猜测每一个bounding box的注意对应两个glimpse
# 节点注意力权向量为a，a是[0, 1]^36，a_i是第i个节点的权重


class NoOpModule(nn.Module):
    """
    NoOp模块遵循Explainable Neural Computation via Stack Neural Module Networks设置
    用来补齐剩余的操作空缺
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        # 无操作，直接输出
        return att_stack, stack_ptr, mem_in


class AndModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        att2 = _read_from_stack(att_stack, stack_ptr)

        # 这里写入的是torch.zeros(feat.size(0), feat.size(1), 1)，即维度为(batch_size, 36, 1)的零向量
        # 在栈指针位置写入零
        att_stack = _write_to_stack(att_stack, stack_ptr, torch.zeros(feat.size(0), feat.size(1), 1).to(feat.device))

        stack_ptr = _move_ptr_bw(stack_ptr)

        att1 = _read_from_stack(att_stack, stack_ptr)

        # and模块取a_1和a_2的最小值
        att_out = torch.min(att1, att2)

        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        # And模块弹出两个值，写入一个值
        return att_stack, stack_ptr, mem_in.clone().zero_()


class FindModule(nn.Module):
    # Find是AttendNode
    # 这里遵循Explainable and Explicit Visual Reasoning over Scene Graphs的Sec3.3.1 Attention Functions
    def __init__(self, **kwargs):
        """
        dim_hidden, # hidden of seq2seq 默认1024
        dim_v, # vertex and edge embedding of scene graph 默认是512
        glimpses, 默认2
        dropout_prob, 默认0.5
        """
        super().__init__()
        self.map_c = nn.Linear(kwargs['dim_hidden'], kwargs['dim_v'])
        self.x_conv = nn.Linear(kwargs['dim_v'], kwargs['glimpses'])
        self.drop = nn.Dropout(kwargs['dropout_prob'])
        self.fusion = Fusion()

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        # c_i: 第t个时间步的文本参数，维度为(batch_size, dim_lstm(即dim_hidden))
        # 将文本参数过一层线性层，
        query = self.map_c(self.drop(c_i)).unsqueeze(1)  # (batch_size, 1, dim_v)

        # feat维度：(batch_size, 36, 512)
        x = self.fusion(feat, query)

        att_out = self.x_conv(self.drop(x))  # (batch_size, num_feat, glimpse)

        # 在每个glimpse上对36个值求softmax，纵向就是节点注意了
        att_out = F.softmax(att_out, dim=1)  # (batch_size, num_feat, glimpse)

        # att_out = torch.sigmoid(att_out)
        # 前移堆栈指针
        stack_ptr = _move_ptr_fw(stack_ptr)

        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, mem_in.clone().zero_()


class FilterModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.Find = FindModule(**kwargs)
        self.And = AndModule(**kwargs)

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        # 遵循Explainable and Explicit Visual Reasoning over Scene Graphs的Sec3.3.2 Composite Reasoning Modules
        # Filter -> And(a, Find(q))
        att_stack, stack_ptr, _ = self.Find(vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr,
                                            mem_in)
        att_stack, stack_ptr, _ = self.And(vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr,
                                           mem_in)
        return att_stack, stack_ptr, mem_in.clone().zero_()


class TransformModule(nn.Module):
    """
    由节点注意向量a和边缘注意矩阵W，沿着注意关系转移节点权值，找到新的对象
    """
    def __init__(self, **kwargs):
        """
        dim_hidden, # hidden of seq2seq 默认1024
        dim_edge, 默认256
        """
        super().__init__()
        self.map_c = nn.Linear(kwargs['dim_hidden'], kwargs['dim_edge'])
        self.map_weight = nn.Linear(kwargs['dim_edge'], 1)
        self.glimpses = kwargs['glimpses']

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        # feat_edge: (batch_size, 36, 36, 256)
        batch_size = feat_edge.size(0)

        # c_i: 第t个时间步的文本参数，维度为(batch_size, dim_lstm(即dim_hidden))
        # 将c_i由1024映射到256
        # (batch_size, 1024) -> (batch_size, 256) -> (batch_size, 1, 1, 256) -> (batch_size, 36, 36, 256)
        query = self.map_c(c_i).view(batch_size, 1, 1, -1).expand_as(feat_edge)

        # 将融合了文本参数的query与边特征相乘
        elt_prod = query * feat_edge

        # 过一层sigmoid，weit_matrix: (batch_size, 36, 36)
        weit_matrix = F.sigmoid(torch.sum(elt_prod, dim=3))

        # 可能是作者认为只有两个bounding box相重叠才可以转移权重（为什么需要bounding box在空间上相接才可以转移权重）
        weit_matrix = weit_matrix * relation_mask.float()

        att_in = _read_from_stack(att_stack, stack_ptr).permute(0, 2, 1)  # (batch_size, glimpse, att_dim)

        att_out = torch.matmul(att_in, weit_matrix).permute(0, 2, 1)  # (batch_size, att_dim, glimpse)

        # 除以最大值归一
        norm = torch.max(att_out, dim=1, keepdim=True)[0].detach()

        norm[norm <= 1] = 1

        att_out /= norm
        # ---------
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, mem_in.clone().zero_()


class DescribeModule(nn.Module):
    """
    K = 1，仅为矩阵相乘
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        batch_size = feat.size(0)

        att_in = _read_from_stack(att_stack, stack_ptr).permute(0, 2, 1)

        # att_in: (batch_size, glimpse, att_dim)
        # vision_feat: (batch_size, 36, 2053)
        mem_out = torch.bmm(att_in, vision_feat)

        mem_out = mem_out.view(batch_size, -1)  # (batch_size, glimpse*dim_vision)
        return att_stack, stack_ptr, mem_out


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return - (x - y) ** 2 + F.relu(x + y)


def _move_ptr_fw(stack_ptr):
    """
    Move the stack pointer forward (i.e. to push to stack).
    stack_ptr: (batch_size, stack_len)
    Return: (batch_size, stack_len)
    向前移动堆栈指针
    遵循Explainable Neural Computation via Stack Neural Module Networks运算方式
    将stack_ptr赋值为：一维卷积(stack_ptr, [0, 0, 1])
    """
    filter_fw = torch.FloatTensor([1, 0, 0]).view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.size()
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack top, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_top_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_top_mask[stack_len - 1] = 1  # [stack_len, ]
    new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _move_ptr_bw(stack_ptr):
    """
    Move the stack pointer backward (i.e. to pop from stack).
    向后移动堆栈指针
    遵循Explainable Neural Computation via Stack Neural Module Networks运算方式
    将stack_ptr赋值为：一维卷积(stack_ptr, [1, 0, 0])
    """

    # filter_fw: tensor([[[0., 0., 1.]]])
    filter_fw = torch.FloatTensor([0, 0, 1]).view(1, 1, 3).to(stack_ptr.device)

    batch_size, stack_len = stack_ptr.size()

    # stack_ptr存了batch_size个栈指针，先扩展一维，每一行对应[1, stack_len]的栈指针向量，再与[0, 0, 1]进行一维卷积
    # 最后转成(batch_size, stack_len)
    # 并且，padding=1，经过[0,0,1]一维卷积，stack_ptr[0]必然为0
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)

    # when the stack pointer is already at the stack bottom, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    # 这里考虑如果已经退到栈底，令stack_bottom_mask为[0,0,0,1]，将原来在栈底的1加回去
    stack_bottom_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_bottom_mask[0] = 1

    new_stack_ptr += stack_bottom_mask * stack_ptr

    return new_stack_ptr


def _read_from_stack(att_stack, stack_ptr):
    """
    Read the value at the given stack pointer.
    给定栈指针，读取指针对应的值

    att_stack维度: (batch_size, 36, 2, 4)
    stack_ptr存batch_size个栈顶指针p，每个p为stack_len维的one hot向量，维度为(batch_size, stack_len(4))
    """
    batch_size, stack_len = stack_ptr.size()

    # 将栈指针的维度扩展为(batch_size, 1, 1, 4)，便于与注意力堆栈相乘
    stack_ptr_expand = stack_ptr.view(batch_size, 1, 1, stack_len)

    # The stack pointer is a one-hot vector, so just do dot product
    # 从最后一维的stack_len元素中取栈指针为1的位置的元素
    # 因为栈指针是一个one hot向量
    # 所以，att维度是(batch_size, att_dim(num_feat), glimpse)，最后一维每一行存了glimpse个元素，每个元素都对应着栈指针为1标记的元素
    att = torch.sum(att_stack * stack_ptr_expand, dim=-1)

    return att  # (batch_size, att_dim(num_feat), glimpse)


def _write_to_stack(att_stack, stack_ptr, att):
    """
    Write value 'att' into the stack at the given stack pointer. Note that the
    result needs to be assigned back to att_stack
    在给定堆栈指针处将值“att”写入堆栈。请注意，结果需要分配回att_stack
    """
    batch_size, stack_len = stack_ptr.size()

    # 将栈指针expand成(batch_size, 1, 1, stack_len)维度，是在与att和att_stack相乘时
    # 利用广播机制，将栈指针拷贝glimpse份，每一个glimpse对应一份
    stack_ptr_expand = stack_ptr.view(batch_size, 1, 1, stack_len)

    # 如果传入的待赋的值缺一个维度，则添加一个维度，保证维度数为4
    if att.dim() == 3:
        att = att.unsqueeze(3)

    # att * stack_ptr_expand存了一份在栈指针指向的位置赋值的结果
    # 与att_stack * (1 - stack_ptr_expand)相加，把赋值的结果加在所有glimpse上
    att_stack = att * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)

    # e.g. 假设num_feat = 2, batch_size = 2

    # att的维度是(2, 2, 1, 1)
    # att = tensor([[[[0.7500]],
    #                [[0.5501]]],
    #               [[[0.0383]],
    #                [[0.3418]]]])

    # stack_ptr_expand的维度是(2, 1, 1, 4)
    # stack_ptr_expand = tensor([[[[1., 0., 0., 0.]]],
    #                           [[[0., 0., 1., 0.]]]])

    # att与stack_ptr_expand相乘，将栈指针指向的元素赋值为待赋值的注意力值
    # att * stack_ptr_expand = tensor([[[[0.7500, 0.0000, 0.0000, 0.0000]],
    #                                   [[0.5501, 0.0000, 0.0000, 0.0000]]],
    #                                  [[[0.0000, 0.0000, 0.0383, 0.0000]],
    #                                   [[0.0000, 0.0000, 0.3418, 0.0000]]]])

    # 在每一个glimpse上都要赋值
    # att_stack * (1 - stack_ptr_expand) = tensor([[[[0.0000, 0.6897, 0.2026, 0.2006],
    #                                                [0.0000, 0.9997, 0.4420, 0.5310]],
    #                                               [[0.0000, 0.9777, 0.4308, 0.1216],
    #                                                [0.0000, 0.8881, 0.8291, 0.9359]]],
    #                                              [[[0.5809, 0.8674, 0.0000, 0.8899],
    #                                                [0.4606, 0.7640, 0.0000, 0.2863]],
    #                                               [[0.0821, 0.1416, 0.0000, 0.5089],
    #                                                [0.3629, 0.8481, 0.0000, 0.2991]]]])

    # 然后将上面两个结果相加，每一个glimpse上的对应栈指针指向的位置都赋值为att
    #      纵向是num_feat，图像的特征维度，应该为36
    #              |
    #              |
    # tensor([[[[0.7500, 0.6897, 0.2026, 0.2006], <-- glimpse_1，维度是栈深    ---
    #           [0.7500, 0.9997, 0.4420, 0.5310]], <-- glimpse_2                | batch中的一个数据（一张图像）
    #          [[0.5501, 0.9777, 0.4308, 0.1216],                               |
    #           [0.5501, 0.8881, 0.8291, 0.9359]]],                          ---
    #         ---------------------------------------------------
    #         [[[0.5809, 0.8674, 0.0383, 0.8899],
    #           [0.4606, 0.7640, 0.0383, 0.2863]],
    #          [[0.0821, 0.1416, 0.3418, 0.5089],
    #           [0.3629, 0.8481, 0.3418, 0.2991]]]])

    return att_stack  # (batch_size, att_dim, glimpse, stack_len)


def _sharpen_ptr(stack_ptr, hard):
    """
    Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
    or softmax. The stack values should always sum up to one for each instance.
    """
    if hard:
        # hard (non-differentiable) sharpening with argmax
        stack_len = stack_ptr.size(1)
        new_stack_ptr_indices = torch.argmax(stack_ptr, dim=1)[1]
        new_stack_ptr = convert_to_one_hot(new_stack_ptr_indices, stack_len)
    else:
        # soft (differentiable) sharpening with softmax
        temperature = 0.1
        new_stack_ptr = F.softmax(stack_ptr / temperature, dim=1)
    return new_stack_ptr


def _build_module_validity_mat(stack_len, module_names):
    """
    Build a module validity matrix, ensuring that only valid modules will have
    non-zero probabilities. A module is only valid to run if there are enough
    attentions to be popped from the stack, and have space to push into
    (e.g. _Find), so that stack will not underflow or overflow by design.

    module_validity_mat is a stack_len x num_module matrix, and is used to
    multiply with stack_ptr to get validity boolean vector for the modules.

    构建模块有效性矩阵，确保只有有效模块具有非零概率。
    只有当有足够的注意力从堆栈中弹出，并且还有空间推入堆栈（例如，_Find）时，模块才有效运行，
    这样堆栈就不会因设计而下溢或溢出。
    module_validity_mat是一个stack_len x num_module矩阵（4*6），用于与stack_ptr相乘以获得模块的有效布尔向量。
    """

    module_validity_mat = np.zeros((stack_len, len(module_names)), np.float32)
    # 遍历所有定义的模块（MODULE_INPUT_NUM的键值）
    # 有六列，每一列存放对应模块的有效值
    for n_m, m in enumerate(module_names):
        # a module can be run only when stack ptr position satisfies
        # (min_ptr_pos <= ptr <= max_ptr_pos), where max_ptr_pos is inclusive
        # 1) minimum position: 最小的位置
        #    stack need to have MODULE_INPUT_NUM[m] things to pop from 堆栈需要弹出MODULE_INPUT_NUM[m]个元素
        # min_ptr_pos保证这个模块运算输入的需要
        min_ptr_pos = MODULE_INPUT_NUM[m]
        # 堆栈指针差值：MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m]
        # 保证指针+差值 <= 4（默认值） - 1 = 3（栈顶）
        # the stack ptr diff=(MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m])
        # ensure that ptr + diff <= stack_len - 1 (stack top)
        # max_ptr_pos保证经过这个模块的运算后堆栈不会溢出
        max_ptr_pos = (
                stack_len - 1 + MODULE_INPUT_NUM[m] - MODULE_OUTPUT_NUM[m])
        # 在每一列有效栈深的序号处置一
        module_validity_mat[min_ptr_pos:max_ptr_pos + 1, n_m] = 1.

    return module_validity_mat

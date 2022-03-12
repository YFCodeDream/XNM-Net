# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from IPython import embed


def invert_dict(d):
    # 将字典的键和键值反转
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        # 在原有vocab的基础上加以下由相应键值对反转键值的键值对
        # 由顺序编码返回对应的token
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    return vocab


class VQADataset(Dataset):
    def __init__(self,
                 answers,
                 questions,
                 questions_len,
                 q_image_indices,
                 feature_h5,
                 feat_coco_id_to_index,
                 num_answer,
                 use_spatial):

        # 存储进pt文件的内容如下：
        # questions：转成np.array格式的填充完毕的question每一个token编码转成的列表，填充长度是最大的问题编码序列长度
        # questions_len：每一个question的长度的列表也转换为np.array格式
        # image_idxs：每一个annotation对应的图像id
        # answers：每一个annotation有多个answer，每个answer对应一个编码，组成对应的编码列表，即answers是一个二维列表
        # glove：question出现的所有token的对应glove embedding向量

        # feature.h5
        # features变量维度为（图像张数，resnet101编码显著图像区域特征维度（2048），采用前36个特征（36））
        # coco_ids变量是一维向量，向量长度为图像张数，每一张图像对应一个id
        # boxes变量的维度为（图像张数，4（x1, y1, x2, y2），36）
        # width和height同样为一维向量，向量长度为图像张数，每一张图像对应一个width与height

        # num_answer是vocab['answer_token_to_idx']的长度，即出现过的所有answer的个数
        # （这里取了前3000个出现次数最多的answer，所以num_answer=3000)

        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_q_image_idxs = torch.LongTensor(np.asarray(q_image_indices))

        self.feature_h5 = feature_h5
        self.feat_coco_id_to_index = feat_coco_id_to_index
        self.num_answer = num_answer
        self.use_spatial = use_spatial

    def __getitem__(self, index):
        # 重写迭代时候返回对象的方法
        # 如果all_answers非空，从all_answers中取第index个元素，否则置为None
        answer = self.all_answers[index] if self.all_answers is not None else None
        # answer是第index个元素对应的annotation里的answer编码列表
        if answer is not None:
            # 如果answer非空（应该都是非空，即使answer为空列表）
            _answer = torch.zeros(self.num_answer)
            # 以answer包含的编码为索引，该索引处对应的answer每出现一次，就加一，其余为0
            for i in answer:
                _answer[i] += 1
            # 将answer赋值为生成的出现次数的统计向量
            answer = _answer

        # 问题是all_questions的第index个
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]

        image_idx = self.all_q_image_idxs[index].item()  # coco_id

        # fetch vision features 获取视觉特征
        # 依据coco_id找到对应的索引
        index = self.feat_coco_id_to_index[image_idx]

        # 从feature.h5文件中找到相应的图片数据
        with h5py.File(self.feature_h5, 'r') as f:
            # 这里的index是feature.h5里存的按顺序的索引，不是coco_id，是由coco_id转来的索引
            # vision_feat是2048*36
            vision_feat = f['features'][index]
            # boxes是4*36
            boxes = f['boxes'][index]
            # w和h是标量
            w = f['widths'][index]
            h = f['heights'][index]

        # 应该是文章中写的使用两个节点之间坐标的差值来表示edge的编码
        # len(boxes[0])是36，spatial_feat是5*36
        spatial_feat = np.zeros((5, len(boxes[0])))
        spatial_feat[0, :] = boxes[0, :] * 2 / w - 1  # x1
        spatial_feat[1, :] = boxes[1, :] * 2 / h - 1  # y1
        spatial_feat[2, :] = boxes[2, :] * 2 / w - 1  # x2
        spatial_feat[3, :] = boxes[3, :] * 2 / h - 1  # y2
        # 这个应该是bounding box的面积
        spatial_feat[4, :] = (spatial_feat[2, :] - spatial_feat[0, :]) * (spatial_feat[3, :] - spatial_feat[1, :])

        if self.use_spatial:
            # 如果使用空间特征，即将刚刚的空间特征拼接在视觉特征上
            # 拼接之后就是（2048+5）*36 = 2053*36
            vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)
        vision_feat = torch.from_numpy(vision_feat).float()

        # 纵向是特征数36
        num_feat = boxes.shape[1]
        # relation_mask是36*36的矩阵
        relation_mask = np.zeros((num_feat, num_feat))
        for i in range(num_feat):
            for j in range(i + 1, num_feat):
                # if there is no overlap between two bounding box
                # 如果两个bounding box没有重叠
                if boxes[0, i] > boxes[2, j] or \
                        boxes[0, j] > boxes[2, i] or \
                        boxes[1, i] > boxes[3, j] or \
                        boxes[1, j] > boxes[3, i]:
                    pass
                else:
                    # 否则记relation_mask对称矩阵的ij和ji均为1
                    relation_mask[i, j] = relation_mask[j, i] = 1
        relation_mask = torch.from_numpy(relation_mask).byte()

        # 所以，VQADataset依据每一个index返回：
        # image_idx：图像的coco_id
        # answer：出现次数的统计向量，维度是vocab['answer_token_to_idx']的长度（3000）
        # question：转成np.array格式的填充完毕的question每一个token编码转成的列表，填充长度是最大的问题编码序列长度
        # questions_len：转换为np.array格式的每一个question的长度的列表
        # vision_feat：如果使用空间特征，就把对应的x1,y1,x2,y2,bounding box面积拼接在2048维特征后，变成2053维
        # relation_mask：36个bounding box是否有重叠的标记矩阵，有重叠则在对应位置记为1，为对称矩阵
        return image_idx, answer, question, question_len, vision_feat, relation_mask

    def __len__(self):
        # 返回所有问题的个数
        return len(self.all_questions)


class VQADataLoader(DataLoader):
    def __init__(self, **kwargs):
        # vocab.json由preprocess_questions.py生成
        # 取出vocab.json的地址
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % vocab_json_path)
        # vocab里存了question answer program
        # vocab = {
        #     'question_token_to_idx': question_token_to_idx,
        #     'answer_token_to_idx': answer_token_to_idx,
        #     # 编码模块程序
        #     'program_token_to_idx': {token: i for i, token in
        #                              enumerate(['<eos>', 'find', 'relate', 'describe', 'is', 'and'])}
        # }
        vocab = load_vocab(vocab_json_path)

        # 取出之前生成的train_question.pt/val_question.pt/test_question.pt文件地址
        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % question_pt_path)
        with open(question_pt_path, 'rb') as f:
            # 用pickle加载模型文件
            obj = pickle.load(f)

            # 从question.pt里拿到相关的数据
            # 存储进pt文件的内容如下：
            # questions：转成np.array格式的填充完毕的question每一个token编码转成的列表，填充长度是最大的问题编码序列长度
            # questions_len：每一个question的长度的列表也转换为np.array格式
            # image_idxs：每一个annotation对应的图像id
            # answers：每一个annotation有多个answer，每个answer对应一个编码，组成对应的编码列表
            # glove：question出现的所有token的对应glove embedding向量
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_image_indices = obj['image_idxs']
            answers = obj['answers']
            glove_matrix = obj['glove']

        use_spatial = kwargs.pop('spatial')

        # feature.h5
        # features变量维度为（图像张数，resnet101编码显著图像区域特征维度（2048），采用前36个特征（36））
        # coco_ids变量是一维向量，向量长度为图像张数，每一张图像对应一个id
        # boxes变量的维度为（图像张数，4（x1, y1, x2, y2），36）
        # width和height同样为一维向量，向量长度为图像张数，每一张图像对应一个width与height

        with h5py.File(kwargs['feature_h5'], 'r') as features_file:
            # 从feature.h5中读取coco_ids
            coco_ids = features_file['ids'][()]
        # 存储coco_id：索引的字典，由coco_id转成对应的索引
        feat_coco_id_to_index = {idx: i for i, idx in enumerate(coco_ids)}
        self.feature_h5 = kwargs.pop('feature_h5')
        self.dataset = VQADataset(answers,
                                  questions,
                                  questions_len,
                                  q_image_indices,
                                  self.feature_h5,
                                  feat_coco_id_to_index,
                                  len(vocab['answer_token_to_idx']),
                                  use_spatial)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    # noinspection PyTypeChecker
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

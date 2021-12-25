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
        # convert data to tensor 将数据转换为tensor
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
        if answer is not None:
            # 如果answer非空
            _answer = torch.zeros(self.num_answer)
            # 以answer包含的元素为索引，该索引处为1，其余为0
            for i in answer:
                _answer[i] += 1
            # 将answer赋值为生成的0-1向量
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
            vision_feat = f['features'][index]
            boxes = f['boxes'][index]
            w = f['widths'][index]
            h = f['heights'][index]

        # 应该是文章中写的使用两个节点之间坐标的差值来表示edge的编码
        spatial_feat = np.zeros((5, len(boxes[0])))
        spatial_feat[0, :] = boxes[0, :] * 2 / w - 1  # x1
        spatial_feat[1, :] = boxes[1, :] * 2 / h - 1  # y1
        spatial_feat[2, :] = boxes[2, :] * 2 / w - 1  # x2
        spatial_feat[3, :] = boxes[3, :] * 2 / h - 1  # y2
        # 这个应该是bounding box的面积
        spatial_feat[4, :] = (spatial_feat[2, :] - spatial_feat[0, :]) * (spatial_feat[3, :] - spatial_feat[1, :])

        if self.use_spatial:
            # 如果使用空间特征，即将刚刚的空间特征拼接在视觉特征上
            vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)
        vision_feat = torch.from_numpy(vision_feat).float()

        # 纵向是特征数
        num_feat = boxes.shape[1]
        relation_mask = np.zeros((num_feat, num_feat))
        for i in range(num_feat):
            for j in range(i + 1, num_feat):
                # if there is no overlap between two bounding box 如果两个bounding box没有重叠
                if boxes[0, i] > boxes[2, j] or \
                        boxes[0, j] > boxes[2, i] or \
                        boxes[1, i] > boxes[3, j] or \
                        boxes[1, j] > boxes[3, i]:
                    pass
                else:
                    # 否则记relation_mask对称矩阵的ij和ji均为1
                    relation_mask[i, j] = relation_mask[j, i] = 1
        relation_mask = torch.from_numpy(relation_mask).byte()

        return image_idx, answer, question, question_len, vision_feat, relation_mask

    def __len__(self):
        return len(self.all_questions)


class VQADataLoader(DataLoader):
    def __init__(self, **kwargs):
        # vocab.json由preprocess_questions.py生成
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % vocab_json_path)
        # vocab里存了question answer program
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % question_pt_path)
        with open(question_pt_path, 'rb') as f:
            # 用pickle加载模型文件
            obj = pickle.load(f)

            # 从question.pt里拿到相关的数据
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_image_indices = obj['image_idxs']
            answers = obj['answers']
            glove_matrix = obj['glove']

        use_spatial = kwargs.pop('spatial')
        with h5py.File(kwargs['feature_h5'], 'r') as features_file:
            # 从feature.h5中读取coco_ids
            coco_ids = features_file['ids'][()]
        # 存储coco_id：索引的字典
        feat_coco_id_to_index = {idx: i for i, idx in enumerate(coco_ids)}
        self.feature_h5 = kwargs.pop('feature_h5')
        self.dataset = VQADataset(answers, questions, questions_len, q_image_indices,
                                  self.feature_h5, feat_coco_id_to_index, len(vocab['answer_token_to_idx']),
                                  use_spatial)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    # noinspection PyTypeChecker
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

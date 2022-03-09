#!/usr/bin/env python3
import re
import os
import argparse
import json
import numpy as np
import pickle
from utils import encode
from collections import Counter

"""
Preprocessing script for VQA question files.
VQA问题文件的预处理
"""

# according to https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/data.py
_special_chars = re.compile('[^a-z0-9 ]*')
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def process_punctuation(s):
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()


def main(args):
    print('Loading data')
    annotations, questions = [], []
    if args.input_annotations_json is not None:
        # 这里改动一下，分隔符改成+，不然绝对路径有盘符:，会报错
        for f in args.input_annotations_json.split('+'):
            # v2_mscoco_train2014_annotations.json的格式
            # {
            # "info" : info,
            # "data_type": str,
            # "data_subtype": str,
            # "annotations" : [annotation],
            # "license" : license
            # }
            #
            # info {
            # "year" : int,
            # "version" : str,
            # "description" : str,
            # "contributor" : str,
            # "url" : str,
            # "date_created" : datetime
            # }
            #
            # license{
            # "name" : str,
            # "url" : str
            # }
            #
            # annotation{
            # "question_id" : int,
            # "image_id" : int,
            # "question_type" : str,
            # "answer_type" : str,
            # "answers" : [answer],
            # "multiple_choice_answer" : str
            # }
            #
            # answer{
            # "answer_id" : int,
            # "answer" : str,
            # "answer_confidence": str
            # }
            # data_type: source of the images (mscoco or abstract_v002).
            # data_subtype: type of data subtype
            #   (e.g. train2014/val2014/test2015 for mscoco, train2015/val2015 for abstract_v002).
            # question_type: type of the question determined by the first few words of the question.
            #   For details, please see README.
            # answer_type: type of the answer. Currently, "yes/no", "number", and "other".
            # multiple_choice_answer: most frequent ground-truth answer.
            # answer_confidence: subject's confidence in answering the question.
            #   For details, please see Antol et al., ICCV 2015.
            annotations += json.load(open(f, 'r'))['annotations']
            # annotations里存放着annotation，每一个annotation就是：
            # annotation{
            # "question_id" : int,
            # "image_id" : int,
            # "question_type" : str,
            # "answer_type" : str,
            # "answers" : [answer],
            # "multiple_choice_answer" : str
            # }
    # 这里改动一下，分隔符改成+，不然绝对路径有盘符:，会报错
    for f in args.input_questions_json.split('+'):
        # v2_OpenEnded_mscoco_train2014_questions.json的格式
        # {
        #     "info": info,
        #     "task_type": str,
        #     "data_type": str,
        #     "data_subtype": str,
        #     "questions": [question],
        #     "license": license
        # }
        #
        # info
        # {
        #     "year": int,
        #     "version": str,
        #     "description": str,
        #     "contributor": str,
        #     "url": str,
        #     "date_created": datetime
        # }
        #
        # license
        # {
        #     "name": str,
        #     "url": str
        # }
        #
        # question
        # {
        #     "question_id": int,
        #     "image_id": int,
        #     "question": str
        # }
        # task_type: type of annotations in the JSON file (OpenEnded).
        # data_type: source of the images (mscoco or abstract_v002).
        # data_subtype: type of data subtype
        # (e.g. train2014/val2014/test2015 for mscoco, train2015/val2015 for abstract_v002).
        questions += json.load(open(f, 'r'))['questions']
        # questions里有多个question，每一个question：
        # question
        # {
        #     "question_id": int,
        #     "image_id": int,
        #     "question": str
        # }
    # 获取question的个数
    print('number of questions: %s' % len(questions))
    # 建立字典，键值是question的id，值是question具体问题
    question_id_to_str = {q['question_id']: q['question'] for q in questions}
    if args.mode != 'test':
        assert len(annotations) > 0

    # Either create the vocab or load it from disk
    if args.mode == 'train':
        # 传入的mode为train时，输出字典vocab.json以及train_questions.pt
        print('Building vocab')
        answer_cnt = {}
        # 遍历每一个annotation
        for ann in annotations:
            # 从answers中取每一个answer
            # answer{
            # "answer_id" : int,
            # "answer" : str,
            # "answer_confidence": str
            # }
            answers = [_['answer'] for _ in ann['answers']]
            for i, answer in enumerate(answers):
                answer = process_punctuation(answer)
                # 对于一个特定的answer，answer_cnt记录其出现的次数
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1
                # 将answers这个列表的所有元素重新赋值为经process_punctuation处理过的answer
                answers[i] = answer
            # 将annotations里存的answer全部换成处理后的answer
            ann['answers'] = answers  # update
        answer_token_to_idx = {}
        # 选取前answer_top个出现次数多的answer
        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            # 将该answer编入answer_token_to_idx字典中，对应的值按顺序编号
            # 编号从0开始逐渐递增，越靠前表示出现次数越多
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        # 遍历question的id及其对应问题
        for i, q in question_id_to_str.items():
            # 把问题最后的问号去掉
            question = q.lower()[:-1]
            # 将问题的特殊字符全部去掉
            question = _special_chars.sub('', question)
            # 将question_id: question的字典重新赋值成处理完特殊符号后的question
            question_id_to_str[i] = question
            # 遍历每个question的单词token
            for token in question.split(' '):
                # 将question中所有的token按检测顺序编号，从0开始
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            # 编码模块程序
            'program_token_to_idx': {token: i for i, token in
                                     enumerate(['<eos>', 'find', 'relate', 'describe', 'is', 'and'])}
        }

        print('Write into %s' % args.vocab_json)
        # 把vocab写入vocab.json
        with open(args.vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        #
        print('Loading vocab')
        with open(args.vocab_json, 'r') as f:
            vocab = json.load(f)
        for ann in annotations:
            answers = [_['answer'] for _ in ann['answers']]
            for i, answer in enumerate(answers):
                answer = process_punctuation(answer)
                answers[i] = answer
            ann['answers'] = answers  # update
        for i, q in question_id_to_str.items():
            question = q.lower()[:-1]
            question = _special_chars.sub('', question)
            question_id_to_str[i] = question

    # Encode all questions 编码所有的question
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    image_idxs = []
    answers = []
    if args.mode in {'train', 'val'}:
        # 如果为训练或者验证模式
        for a in annotations:
            # question_id_to_str：question_id: 处理完特殊符号后的question
            question = question_id_to_str[a['question_id']]
            # 分割question的token词元
            question_tokens = question.split(' ')
            # 依据vocab.json里的question中所有token的顺序编码，把question变成编号序列，存储在question_encoded
            question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            # 每一个question的编号序列存在questions_encoded列表里
            questions_encoded.append(question_encoded)
            # 编号序列的长度存在questions_len列表里
            questions_len.append(len(question_encoded))
            # 对应的图像id存在image_idxs里
            image_idxs.append(a['image_id'])

            answer = []
            for per_ans in a['answers']:
                # 如果处理后的answer在vocab['answer_token_to_idx']的键值里
                if per_ans in vocab['answer_token_to_idx']:
                    # 取出当前answer对应的编码
                    i = vocab['answer_token_to_idx'][per_ans]
                    # 将编码存储在answer列表里
                    answer.append(i)
            # 将answer对应的编码列表存储在answers里
            answers.append(answer)
    elif args.mode == 'test':
        for q in questions:  # remain the original order to match the question_id
            question = question_id_to_str[q['question_id']]  # processed question
            question_tokens = question.split(' ')
            question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            image_idxs.append(q['image_id'])
            answers.append([0])

    # Pad encoded questions 填充question转成的编码序列
    # 取最大的question编码长度
    max_question_length = max(len(x) for x in questions_encoded)
    # 如果question的编码序列没有达到最大长度，则用NULL对应的编码填充（用0填充）
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    # 将填充完毕的question编码转成np.array格式
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    # 每一个question的长度的列表也转换为np.array格式
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if args.mode == 'train':
        # 反转vocab['question_token_to_idx']，变成idx: token
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        # 读取转换的glove的pickle文件
        glove = pickle.load(open(args.glove_pt, 'rb'))
        # 取每一个词向量的维度
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            # 如果查找到对应的glove embedding的向量，就赋值给vector，否则赋值为维度为词向量维度的全零向量
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            # 将question出现的所有token的对应glove embedding向量存入glove_matrix中
            glove_matrix.append(vector)
            print()
        # glove_matrix转成np.array
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing')
    # 存储进pt文件的内容如下：
    # questions：转成np.array格式的填充完毕的question每一个token编码转成的列表，填充长度是最大的问题编码序列长度
    # questions_len：每一个question的长度的列表也转换为np.array格式
    # image_idxs：每一个annotation对应的图像id
    # answers：每一个annotation有多个answer，每个answer对应一个编码，组成对应的编码列表
    # glove：question出现的所有token的对应glove embedding向量
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'image_idxs': np.asarray(image_idxs),
        'answers': answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_top', default=3000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors '
                             'represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--input_questions_json', required=True)
    parser.add_argument('--input_annotations_json', help='not need for test mode')
    parser.add_argument('--output_pt', required=True)
    parser.add_argument('--vocab_json', required=True)
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    main(args)

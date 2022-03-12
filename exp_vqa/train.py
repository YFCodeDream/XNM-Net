import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 自定义的DataLoader
from DataLoader import VQADataLoader
from model.net import XNMNet
from utils.misc import todevice
from validate import validate

# C:\Users\yfcod\Desktop\项目代码\XNM-Net 项目根目录加入系统变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # to import shared utils

# 配置logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


# noinspection PyUnresolvedReferences
def train(args):
    logging.info("Create train_loader and val_loader.........")
    # 训练的时候的DataLoader的参数
    train_loader_kwargs = {
        'question_pt': args.train_question_pt,
        'vocab_json': args.vocab_json,
        'feature_h5': args.feature_h5,
        'batch_size': args.batch_size,
        'spatial': args.spatial,
        'num_workers': 2,
        'shuffle': True
    }
    # 实例化VQADataLoader为训练的DataLoader
    train_loader = VQADataLoader(**train_loader_kwargs)
    if args.val:
        # 如果参数中设置了val键值，则实例化VQADataLoader为验证的DataLoader
        val_loader_kwargs = {
            'question_pt': args.val_question_pt,
            'vocab_json': args.vocab_json,
            'feature_h5': args.feature_h5,
            'batch_size': args.batch_size,
            'spatial': args.spatial,
            'num_workers': 2,
            'shuffle': False
        }
        val_loader = VQADataLoader(**val_loader_kwargs)

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 模型参数
    model_kwargs = {
        'vocab': train_loader.vocab,
        'dim_v': args.dim_v,
        'dim_word': args.dim_word,
        'dim_hidden': args.dim_hidden,
        'dim_vision': args.dim_vision,
        'dim_edge': args.dim_edge,
        'cls_fc_dim': args.cls_fc_dim,
        'dropout_prob': args.dropout,
        'T_ctrl': args.T_ctrl,
        'glimpses': args.glimpses,
        'stack_len': args.stack_len,
        'device': device,
        'spatial': args.spatial,
        'use_gumbel': args.module_prob_use_gumbel == 1,
        'use_validity': args.module_prob_use_validity == 1,
    }
    # 除了vocab键值对，另存一份副本至model_kwargs_tosave
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}

    # 实例化模型
    model = XNMNet(**model_kwargs).to(device)

    logging.info(model)
    logging.info('load glove vectors')

    # 把train_loader的glove_matrix转换为指定设备的FloatTensor
    train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)

    # 将模型的token的embedding的权重设置为预训练的glove权重
    model.token_embedding.weight.data.set_(train_loader.glove_matrix)
    ################################################################

    # 选出模型中require_grad的参数
    parameters = [p for p in model.parameters() if p.requires_grad]

    # 使用Adam优化器
    optimizer = optim.Adam(parameters, args.lr, weight_decay=0)

    start_epoch = 0
    if args.restore:
        # 如果从指定checkpoint恢复
        print("Restore checkpoint and optimizer...")
        # 找到模型保存的路径，由命令行参数save_dir传入
        ckpt = os.path.join(args.save_dir, 'model.pt')
        # 加载checkpoint的模型
        ckpt = torch.load(ckpt, map_location={'cuda:0': 'cpu'})
        # 迭代轮数更新为检查点保存的epoch数加一
        start_epoch = ckpt['epoch'] + 1

        # 从checkpoint保存的模型参数恢复model和优化器
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # 使用指数衰减的学习率
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / args.lr_halflife))

    logging.info("Start training........")
    # 从之前计算得到的起始轮数起，迭代至命令行参数的num_epoch
    for epoch in range(start_epoch, args.num_epoch):
        # 调用XNMNet的train方法，表明是训练模式
        model.train()

        # 从train_loader加载batch数据
        for i, batch in enumerate(train_loader):
            # progress记录当前的轮数
            progress = epoch + i / len(train_loader)

            # 每一个batch的数据，第一个元素是coco_ids，第二个元素是answers，之后的元素是batch_input
            # batch_input包含：

            # question：转成np.array格式的填充完毕的question每一个token编码转成的列表，填充长度是最大的问题编码序列长度
            # questions_len：转换为np.array格式的每一个question的长度的列表
            # vision_feat：如果使用空间特征，就把对应的x1,y1,x2,y2,bounding box面积拼接在2048维特征后，变成2053维
            # relation_mask：36个bounding box是否有重叠的标记矩阵，有重叠则在对应位置记为1，为对称矩阵

            # 对应XNMNet的forward函数需要的参数
            coco_ids, answers, *batch_input = [todevice(x, device) for x in batch]

            # logits是返回的概率向量
            logits, others = model(*batch_input)

            # 计算NLLLoss（为什么这里不直接使用CrossEntropyLoss呢）
            nll = -nn.functional.log_softmax(logits, dim=1)
            loss = (nll * answers / 10).sum(dim=1).mean()

            # scheduler.step()对learning rate进行调整
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()

            # 剪裁梯度，防止梯度爆炸
            nn.utils.clip_grad_value_(parameters, clip_value=0.5)
            # optimizer.step()进行参数更新，在loss.backward()之后
            optimizer.step()

            # 指定轮数时打印训练信息
            if (i + 1) % (len(train_loader) // 50) == 0:
                logging.info("Progress %.3f  ce_loss = %.3f" % (progress, loss.item()))

        # 保存checkpoint
        save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(args.save_dir, 'model.pt'))
        logging.info(' >>>>>> save to %s <<<<<<' % args.save_dir)

        if args.val:
            # 如果指定了验证集，则开始验证，并打印验证信息
            valid_acc = validate(model, val_loader, device)
            logging.info('\n ~~~~~~ Valid Accuracy: %.4f ~~~~~~~\n' % valid_acc)


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    # 保存模型训练轮数，模型参数，优化器，以及其他参数
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    # input and output 输入输出参数，包括路径参数，训练模型文件名等
    parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--input_dir', required=True)

    parser.add_argument('--train_question_pt', default='train_questions.pt')
    parser.add_argument('--val_question_pt', default='val_questions.pt')

    parser.add_argument('--vocab_json', default='vocab.json')
    parser.add_argument('--feature_h5', default='trainval_feature.h5')

    parser.add_argument('--restore', action='store_true')
    # training parameters 训练参数，包括学习率，迭代轮数，批次大小，是否验证等
    parser.add_argument('--lr', default=8e-4, type=float)
    parser.add_argument('--lr_halflife', default=50000, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--val', action='store_true', help='whether validate after each training epoch')
    # model hyperparameters 模型超参数
    parser.add_argument('--dim_word', default=300, type=int, help='word embedding')
    parser.add_argument('--dim_hidden', default=1024, type=int, help='hidden state of seq2seq parser')

    parser.add_argument('--dim_v', default=512, type=int, help='node embedding')
    parser.add_argument('--dim_edge', default=256, type=int, help='edge embedding')

    parser.add_argument('--dim_vision', default=2048, type=int)
    parser.add_argument('--cls_fc_dim', default=1024, type=int, help='classifier fc dim')
    parser.add_argument('--glimpses', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--T_ctrl', default=3, type=int, help='controller decode length')
    parser.add_argument('--stack_len', default=4, type=int, help='stack length')
    # store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False
    parser.add_argument('--spatial', action='store_true')
    parser.add_argument('--module_prob_use_gumbel', default=0, choices=[0, 1], type=int,
                        help='whether use gumbel softmax for module prob. 0 not use, 1 use')
    parser.add_argument('--module_prob_use_validity', default=1, choices=[0, 1], type=int,
                        help='whether validate module prob.')
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not args.restore:
        # 如果restore为False，则创建save_dir指定的文件夹
        os.mkdir(args.save_dir)
    else:
        # 如果restore为True，判断save_dir是否为文件夹
        assert os.path.isdir(args.save_dir)

    # 配置logger
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # args display 打印参数
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    # concat obsolute path of input files 拼接输入文件路径
    args.train_question_pt = os.path.join(args.input_dir, args.train_question_pt)
    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.feature_h5 = os.path.join(args.input_dir, args.feature_h5)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 如果使用空间特征，则将2048加上5，5对应process_feature里的空间特征
    if args.spatial:
        args.dim_vision += 5

    train(args)


if __name__ == '__main__':
    main()

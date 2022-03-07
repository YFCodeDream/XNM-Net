# According to [https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/preprocess-features.py]

import sys
import argparse
import base64
import os
import csv
import itertools

csv.field_size_limit(sys.maxsize)

import h5py
import torch.utils.data
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_h5', required=True)
    parser.add_argument('--input_tsv_folder', required=True, help='path to trainval_36 or test2015_36')
    parser.add_argument('--test', action='store_true', help='specified when processing test2015_36')
    args = parser.parse_args()
    assert os.path.isdir(args.input_tsv_folder)

    # 图像id，宽，高，边界框数量，边界框，特征
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

    features_shape = (
        # train：82783；val：40504；test：81434，这里是VQA2.0数据集的训练，验证和测试图像数量
        82783 + 40504 if not args.test else 81434,  # number of images in trainval or in test
        # resnet101编码图像显著区域特征维度为2048
        2048,  # dim_vision,
        # 每张图像仅仅采用前36个特征
        36,  # 36 for fixed case, 100 for the adaptive case
    )

    boxes_shape = (
        features_shape[0],
        # 这里的4是x1, y1, x2, y2
        4,
        36,
    )

    path = args.output_h5
    with h5py.File(path, libver='latest') as fd:
        # features变量维度为（图像张数，resnet101编码显著图像区域特征维度（2048），采用前36个特征（36））
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        # coco_ids变量是一维向量，向量长度为图像张数，每一张图像对应一个id
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        # boxes变量的维度为（图像张数，4（x1, y1, x2, y2），36）
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        # width和height同样为一维向量，向量长度为图像张数，每一张图像对应一个width与height
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        readers = []
        for filename in os.listdir(args.input_tsv_folder):
            if not '.tsv' in filename:
                continue
            # 获取解压后的每一个存储了图像特征的tsv文件
            # （每个tsv文件里应该存着36行，每一行是一个图像特征）
            full_filename = os.path.join(args.input_tsv_folder, filename)
            fd = open(full_filename, 'r')
            reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)
            # 把每一张图片存储的tsv特征存入列表中
            readers.append(reader)

        # itertools.chain.from_iterable将多个迭代器连接起来
        reader = itertools.chain.from_iterable(readers)
        for i, item in enumerate(tqdm(reader, total=features_shape[0])):
            coco_ids[i] = int(item['image_id'])
            widths[i] = int(item['image_w'])
            heights[i] = int(item['image_h'])

            buf = base64.decodebytes(item['features'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            # 这里的array的shape应该是（2048，36），正好符合features的存储大小，i表示第i张图像
            array = array.reshape((-1, 2048)).transpose()
            features[i, :, :array.shape[1]] = array

            buf = base64.decodebytes(item['boxes'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            # 同理，这里的array的shape应该是（4，36），符合boxes的存储大小
            array = array.reshape((-1, 4)).transpose()
            boxes[i, :, :array.shape[1]] = array


if __name__ == '__main__':
    main()

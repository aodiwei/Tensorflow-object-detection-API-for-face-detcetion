#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/5/17'
# 
"""
import hashlib
import os

import cv2
import tensorflow as tf

from data_process import dataset_util

image_dir = '/home/model/work/object_detect/WIDER_train/images'

re_height = 300
re_width = 300
resize_image_dir = '../resize_img_{}x{}'.format(re_width, re_height)
is_resize = False

if is_resize:
    if not os.path.exists(resize_image_dir):
        os.mkdir(resize_image_dir)


def parse_sample(filename, f):
    """
    
    :param f:
    :return: 
    """
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)
    poses = []
    truncated = []

    # filename = f.readline().rstrip()
    print(filename)
    filepath = os.path.join(image_dir, filename)
    print(filepath)
    image_raw = cv2.imread(filepath)
    height, width, channel = image_raw.shape
    print("height is %d, width is %d, channel is %d" % (height, width, channel))

    if is_resize:
        image_resize = cv2.resize(image_raw, (re_width, re_height))
        path_pre = os.path.split(filename)[0]
        path = os.path.join(resize_image_dir, path_pre)
        if not os.path.exists(path):
            os.mkdir(path)

        output_img = os.path.join(resize_image_dir, filename)
        cv2.imwrite(output_img, image_resize)
        with open(output_img, 'rb') as ff:
            encoded_image_data = ff.read()
    else:
        with open(filepath, 'rb') as ff:
            encoded_image_data = ff.read()

    key = hashlib.sha256(encoded_image_data).hexdigest()
    face_num = int(f.readline().rstrip())
    valid_face_num = 0

    for i in range(face_num):
        annot = f.readline().rstrip().split()
        # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
        if float(annot[2]) > 25.0:
            if float(annot[3]) > 30.0:
                xmins.append(max(0.005, (float(annot[0]) / width)))
                ymins.append(max(0.005, (float(annot[1]) / height)))
                xmaxs.append(min(0.995, ((float(annot[0]) + float(annot[2])) / width)))
                ymaxs.append(min(0.995, ((float(annot[1]) + float(annot[3])) / height)))
                classes_text.append('face'.encode('utf8'))
                classes.append(1)
                poses.append("front".encode('utf8'))
                truncated.append(int(0))
                print(xmins[-1], ymins[-1], xmaxs[-1], ymaxs[-1], classes_text[-1], classes[-1])
                valid_face_num += 1

    print("Face Number is %d" % face_num)
    print("Valid face number is %d" % valid_face_num)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(int(0)),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    return valid_face_num, tf_example


def wider2tfrecord(path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    with open(path) as f:
        # WIDER FACE DATASET ANNOTATED 12880 IMAGES
        valid_image_num = 0
        invalid_image_num = 0
        # each picture start with filename, use for loop to get filename, other arg use readline fun to read
        for filename in f:
            filename = filename.strip()
            valid_face_number, tf_example = parse_sample(filename, f)
            if valid_face_number != 0:
                writer.write(tf_example.SerializeToString())
                valid_image_num += 1
            else:
                invalid_image_num += 1
                print("Pass!")
    writer.close()

    print("Valid image number is %d" % valid_image_num)
    print("Invalid image number is %d" % invalid_image_num)


if __name__ == '__main__':
    annot = '/home/model/work/object_detect/wider_face_split/wider_face_train_bbx_gt.txt'
    out = 'dataset/wider_face_train_no_resize.record'
    wider2tfrecord(annot, out)
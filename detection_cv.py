#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '5/24/18'
# 
"""
import os

import tensorflow as tf
import cv2

flags = tf.app.flags
flags.DEFINE_string('frozen_pb', 'output/export_ckp/frozen_inference_graph.pb',
                    'Directory frozen_inference_graph.pb which produce by export_inference_graph.py ')

flags.DEFINE_string('input', 'data_process/testpic.jpg',
                    'Directory or file to detect ')

flags.DEFINE_string('output', '',
                    'if not set it will output to input directory')

FLAGS = flags.FLAGS


def detect(sess, img_file):
    """
    
    :param sess: 
    :param img_file: 
    :return: 
    """
    assert img_file.endswith('.jpg') or img_file.endswith('.JPG'), 'img {} is not .jpg'.format(img_file)
    img = cv2.imread(img_file)
    rows = img.shape[0]
    cols = img.shape[1]
    img_re = cv2.resize(img, (300, 300))
    np_img = img_re[:, :, [2, 1, 0]]  # BGR2RGB
    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': np_img.reshape(1, np_img.shape[0], np_img.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.5:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv2.putText(img, '{:.2f}'.format(score), (int(x), int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (125, 255, 51))
            print('img: {}, box No.{}, score: {}'.format(img_file, i, score))
    f = os.path.split(img_file)[-1]
    if FLAGS.output:
        cv2.imwrite(os.path.join(FLAGS.output, f.replace('.jpg', '_box.jpg').replace('.JPG', '_box.jpg')), img)
    else:
        cv2.imwrite(os.path.join(FLAGS.input.replace('.jpg', '_box.jpg').replace('.JPG', '_box.jpg')), img)

    return img


def main(_):
    with tf.gfile.FastGFile(FLAGS.frozen_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        if os.path.isdir(FLAGS.input):
            files = os.listdir(FLAGS.input)
            for f in files:
                try:
                    img_file = os.path.join(FLAGS.input, f)
                    img = detect(sess, img_file)
                except Exception as e:
                    print(e)
        else:
            img = detect(sess, FLAGS.input)

        cv2.imshow('face', img)
        cv2.waitKey()


if __name__ == '__main__':
    tf.app.run()

#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/5/18'
# 
"""
import cv2
import tensorflow as tf

from data_process import dataset_util


def boundingbox(img, boxes, box_type):
    """
    
    :param img: 
    :return: 
    """
    image = cv2.imread(img)
    height, width, channel = image.shape
    print("height is %d, width is %d, channel is %d" % (height, width, channel))

    for (x, y, w, h) in boxes:
        if box_type == 'tfrecord':
            x, y, w, h = format_box_tfrecord2wider(x, y, w, h, image.shape[0], image.shape[1])
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
        elif box_type == 'wider':
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
        else:  # VOC
            x_min, y_min, x_max, y_max = x, y, w, h
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, 'face', (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0))

    cv2.namedWindow('bound', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bound', width, height)
    cv2.imshow('bound', image)
    cv2.waitKey(0)


def format_box_wider2tfrecord(x, y, w, h, real_h, real_w):
    """
    wider to tf_record record the rate of min point and width/height
    :param x:
    :param y:
    :param w:
    :param h:
    :param real_h:
    :param real_w:
    :return:
    """
    print('orig: ', x, y, w, h, real_h, real_w)
    x_ = x / real_w
    y_ = y / real_h
    w_ = (x + w) / real_w
    h_ = (y + h) / real_h

    # return int(x), int(y), int(w), int(h)
    print('rate: ', x_, y_, w_, h_)
    return x_, y_, w_, h_


def format_box_tfrecord2wider(x_, y_, w_, h_, real_h, real_w):
    x = real_w * x_
    y = real_h * y_
    w = real_w * w_ - x
    h = real_h * h_ - y
    print('to orig: ', x, y, w, h, real_h, real_w)
    return int(x), int(y), int(w), int(h)


def add_box_img_arr(win_name, img_arr, boxes):
    """

    :param arr:
    :param boxes:
    :return:
    """
    # cv2.imshow(win_name, img_arr)
    width, height = img_arr.shape[0], img_arr.shape[1]
    for (x, y, w, h) in boxes:
        x, y, w, h = format_box_tfrecord2wider(x, y, w, h, width, height)
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        cv2.rectangle(img_arr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(win_name, width, height)
    cv2.imshow(win_name, img_arr)
    # cv2.waitKey(0)


def show_data_tfrecord(tfrecord_file="../dataset/wider_face_train.record"):
    """

    :type tfrecord_file: object
    :return:
    """
    filename_queue = tf.train.string_input_producer([tfrecord_file])

    with tf.Session() as sess:
        image, label, xmin, ymin, w, h = dataset_util.read_and_decode(filename_queue)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            image_out, label_out, xmin_out, ymin_out, w_out, h_out = sess.run([image, label, xmin, ymin, w, h])
            boxes = zip(xmin_out.values, ymin_out.values, w_out.values, h_out.values)
            add_box_img_arr('pic_{}'.format(i), image_out, boxes)

        coord.request_stop()
        coord.join(threads)

        cv2.waitKey(30000)
        cv2.destroyAllWindows()


def show_data_wider(filename='17_Ceremony_Ceremony_17_325.jpg'):
    """

    :param filename:
    :return:
    """
    bs = [(73, 616, 185, 274),
          (442, 464, 252, 333),
          (795, 481, 213, 269)]
    boundingbox(filename, bs, 'wider')


if __name__ == '__main__':
    show_data_wider()
    # show_data_tfrecord('../dataset/wider_face_train_no_resize.record')

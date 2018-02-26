#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        logging.info('The size of the image is too small!')
        return None

    # My code goes here, fills the dict
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
          os.path.basename(data).encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': dataset_util.bytes_feature(encoded_label),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    }
    #logging.info('feature_dict: %s %d, %d, %s, %s', feature_dict['image/filename'], feature_dict['image/height'], feature_dict['image/width'], feature_dict['image/encoded'], feature_dict['image/label'])
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    logging.info('example: %s', dataset_util.bytes_feature(os.path.basename(data).encode('utf8')))
    return example


def create_tf_record(output_filename, file_pairs):
    # My code goes here
    """Creates a TFRecord file.
    Args:
      output_filename: Path to where output file is saved.

      file_pairs: file pairs to parse and save to tf record.

    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, file_pair in enumerate(list(file_pairs)):
        if idx % 100 == 0 :
            logging.info('On image %d of %d', idx, len(list(file_pairs)))

        image_path = file_pair[0]

        logging.debug('NO %d %s', idx, image_path)

        label_path = file_pair[1]

        logging.debug('NO %d %s', idx, label_path)

        if not os.path.exists(image_path) :
            logging.warning('Could not find %s, ignoring file pair.', image_path)
            continue

        if not os.path.exists(label_path) :
            logging.warning('Could not find %s, ignoring file pair.', label_path)
            continue

        try:
            tf_example = dict_to_tf_example(image_path, label_path)
            if tf_example is not None :
                writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid file pair: %s&%s, ignoring.', image_path, label_path)

    writer.close()



def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):

    #logger = logging.getLogger()

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='/tmp/data8.log',
                    filemode='w')

    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()

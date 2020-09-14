import os
import cv2
import tqdm
import tensorflow as tf
import numpy as np


def image2tfrecords(path_dataset, save_path):
    """
        convert to tfrecord format
        return label_dict, num_classes
    """
    labels_dict = {}
    output_path = os.path.join(save_path, 'data.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    list_datasets = os.listdir(path_dataset)

    for idx_member, name_member in tqdm.tqdm(enumerate(list_datasets)):
        # get path to member folder
        path_to_member = os.path.join(path_dataset, name_member)
        images = os.listdir(path_to_member)
        # get label dict in digit 
        labels_dict[idx_member] = name_member
        for image in images:
            image_path = os.path.join(path_to_member, image)
            np_img = cv2.imread(image_path)
            img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (112,112))
            print(img.shape)
            img_raw = img.tobytes()
            #
            print("len img raw: ", len(img_raw))
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[idx_member]))
            }))
            writer.write(example.SerializeToString())  # Serialize To String
    writer.close()

    return labels_dict, len(list_datasets)

if __name__ == '__main__':

    dataset = '/home/minhdc/Documents/F-Face/Simple_keras/images'
    save = '/home/minhdc/Documents/arc-face-tensorflow/dataset'
    labels_dict, num_classes = image2tfrecords(dataset, save)
    print(num_classes)
    print(labels_dict)
    

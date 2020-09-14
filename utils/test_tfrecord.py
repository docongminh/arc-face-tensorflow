import tensorflow as tf
from utils import parse_function
import os


config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
tfrecords_f = os.path.join('/home/minhdc/Documents/arc-face-tensorflow/dataset', 'data.tfrecords')
dataset = tf.data.TFRecordDataset(tfrecords_f)
dataset = dataset.map(parse_function)
dataset = dataset.shuffle(buffer_size=1)
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
print("next element", next_element)
#begin iteration
for i in range(4):
    sess.run(iterator.initializer)
    images, labels = sess.run(next_element)
    print("type: ", type(images))
    print("len images: ", images.shape)
    print("label: ", labels)
    print("------------------------------")
    # except tf.errors.OutOfRangeError:
        # print("End of dataset")

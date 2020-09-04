import tensorflow as tf
import argparse
import cv2
import numpy as np
from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss
# from nets.L_Resnet_E_IR import get_resnet
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
# from nets.L_Resnet_E_IR_GBN import get_resnet
# from nets.L_Resnet_E_MGPU import get_resnet
# from nets.L_Resnet_E_RBN import get_resnet
import tensorlayer as tl
from verification import ver_test
import time


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file', default='./pre-train/baidu_model_D/InsightFace_iter_best_',
                       type=str, help='the ckpt file path')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--ckpt_index_list',
                        default=['710000.ckpt'], help='ckpt file indexes')
    args = parser.parse_args()
    return args

def imread_img(path, dim=(112, 112)):
    """
        params:
            path: path to image
        return:
            bgr image with shape (112, 112, 3)
    """
    image = cv2.imread(path)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    bgr_img = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    return bgr_img


if __name__ == '__main__':
    args = get_args()
    # ver_list = []
    # ver_name_list = []
    # for db in args.eval_datasets:
    #     print('begin db %s convert.' % db)
    #     data_set = load_bin(db, args.image_size, args)
    #     ver_list.append(data_set)
    #     ver_name_list.append(db)
   
    image = imread_img("images/minh/face_1.jpg")
    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor = net.outputs
    print("embedding: ", embedding_tensor)

    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    # logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    saver = tf.train.Saver()

    # for file_index in args.ckpt_index_list:
    feed_dict_test = {}
    feed_dict_test[images] = image.reshape(1, 112, 112, 3)
    path = args.ckpt_file + args.ckpt_index_list[0]
    saver.restore(sess, path)
    print('ckpt file %s restored!' % args.ckpt_index_list[0])
    # feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
    feed_dict_test[dropout_rate] = 1.0
    t1 = time.time()
    results = sess.run(embedding_tensor, feed_dict_test)
    t2 = time.time()
    # results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
    #                    embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
    #                    input_placeholder=images)
    print("Result: ", results)
    print("max: ", np.argmax(results))
    print("confidence: ", results[0][np.argmax(results)])
    print("Result Vector embedding: ", results.shape)
    
    print("Time Runing: ", t2- t1)
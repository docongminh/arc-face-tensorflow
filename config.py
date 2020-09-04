#
image_height = 112
image_width = 112
chanels = 3
#
net_depth = 50
num_classes = 85164
#
weight_deacy = 5e-4 # learning alg momentum
#
batch_size = 64
buffer_size = 10000
epochs = 100000
learning_rate_steps = [40000, 60000, 80000] # learning rate to train network
schedule_lr_value = [0.001, 0.0005, 0.0003, 0.0001]
momentum = 0.9
#

log_device_mapping = False
summary_path = './summary'
log_file_path = './logs'
saver_maxkeep = 20
show_info_interval = 20 # intervals to save ckpt file
summary_interval = 300 # interval to save summary
ckpt_interval = 10000 # intervals to save ckpt file
tfrecords_file_path = ''
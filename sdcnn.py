#Qingbo, Xin yu
#This script defines some parameters and is the entry.
import os
import tensorflow as tf
from models.model import Model

def main(_):
	#call set_parameter function to intialize the super parameters 
	PARA=set_parameter()
	print(PARA.data_dir)

	#establish the session variable
	sess = tf.Session()
	# establish a model object
	model = Model(sess, PARA)
	#model = Model(sess, set_parameter())
	# act=PARA.action
	# print(act)
	#model.eval(act)()
	getattr(model,PARA.action)()
	#getattr(model, args.option)()

def set_parameter():
	flags = tf.app.flags
	# training
	flags.DEFINE_integer('num_steps', 200, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 10, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model: res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', '/Users/tarus/OnlyInMac/dilated_cnn/pretrain_model/deeplab_resnet_init.ckpt', 
											'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('dilated_type', 'smooth_SSC', 'type of dilated conv: regular, decompose, smooth_GI or smooth_SSC')
	# flags.DEFINE_string('dilated_type', 'SSC', 'type of dilated conv: Basic,Decompose,GI,SSC')
	flags.DEFINE_string('data_list', './dataset/train.txt', 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 2000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 1449, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/val.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', 'output', 'directory for saving outputs')
	flags.DEFINE_integer('test_step', 2000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', 1449, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', './dataset/val.txt', 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', '/Users/tarus/OnlyInMac/dilated_cnn/VOC2012', 'data directory')
	# flags.DEFINE_string('data_dir', '/tempspace2/zwang6/VOC2012', 'data directory')
	flags.DEFINE_integer('batch_size', 8, 'training batch size')
	flags.DEFINE_integer('input_height', 321, 'input image height')
	flags.DEFINE_integer('input_width', 321, 'input image width')
	flags.DEFINE_integer('num_classes', 21, 'number of classes')
	flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
	
	# log
	flags.DEFINE_string('modeldir', 'model', 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')
	flags.DEFINE_string('action', 'train', 'train/test/predict')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	tf.app.run()
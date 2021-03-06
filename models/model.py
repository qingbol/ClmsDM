import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
from .network import *
sys.path.append("..")
from tools import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log, read_labeled_image_list
from tools.image_reader import IMG_MEAN

class Model(object):

	def __init__(self, sess, parameters):
		self.sess = sess
		self.parameters = parameters

	def train(self):
		self.train_setup()

		self.sess.run(tf.global_variables_initializer())

		if self.parameters.pretrain_file is not None:
			self.load(self.loader, self.parameters.pretrain_file)
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
		for step in range(self.parameters.num_steps+1):
			start_time = time.time()
			feed_dict = { self.curr_step : step }

			if step % self.parameters.save_interval == 0:
				loss_value, images, labels, preds, summary, _ = self.sess.run(
					[self.reduced_loss,self.image_batch,self.label_batch,self.pred,self.total_summary,self.train_op],
					feed_dict=feed_dict)
				self.summary_writer.add_summary(summary, step)
				self.save(self.saver, step)
			else:
				loss_value, _ = self.sess.run([self.reduced_loss, self.train_op],
					feed_dict=feed_dict)

			duration = time.time() - start_time
			print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
			write_log('{:d}, {:.3f}'.format(step, loss_value), self.parameters.logfile)

		self.coord.request_stop()
		self.coord.join(threads)

	def test(self):
		self.test_setup()

		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		checkpointfile = self.parameters.modeldir+ '/model.ckpt-' + str(self.parameters.valid_step)
		self.load(self.loader, checkpointfile)

		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		confusion_matrix = np.zeros((self.parameters.num_classes, self.parameters.num_classes), dtype=np.int)
		for step in range(self.parameters.valid_num_steps):
			preds, _, _, c_matrix = self.sess.run([self.pred, self.accu_update_op, self.mIou_update_op, self.parametersusion_matrix])
			confusion_matrix += c_matrix
			if step % 100 == 0:
				print('step {:d}'.format(step))
		print('Pixel Accuracy: {:.3f}'.format(self.accu.eval(session=self.sess)))
		print('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=self.sess)))
		self.compute_IoU_per_class(confusion_matrix)

		self.coord.request_stop()
		self.coord.join(threads)

	def predict(self):
		self.predict_setup()

		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		checkpointfile = self.parameters.modeldir+ '/model.ckpt-' + str(self.parameters.valid_step)
		self.load(self.loader, checkpointfile)

		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		image_list, _ = read_labeled_image_list('', self.parameters.test_data_list)

		for step in range(self.parameters.test_num_steps):
			preds = self.sess.run(self.pred)

			img_name = image_list[step].split('/')[2].split('.')[0]
			im = Image.fromarray(preds[0,:,:,0], mode='L')
			filename = '/%s_mask.png' % (img_name)
			im.save(self.parameters.out_dir + '/prediction' + filename)

			if self.parameters.visual:
				msk = decode_labels(preds, num_classes=self.parameters.num_classes)
				im = Image.fromarray(msk[0], mode='RGB')
				filename = '/%s_mask_visual.png' % (img_name)
				im.save(self.parameters.out_dir + '/visual_prediction' + filename)

			if step % 100 == 0:
				print('step {:d}'.format(step))

		print('The output files has been saved to {}'.format(self.parameters.out_dir))

		self.coord.request_stop()
		self.coord.join(threads)

	def train_setup(self):
		tf.set_random_seed(self.parameters.random_seed)
		
		self.coord = tf.train.Coordinator()
		input_size = (self.parameters.input_height, self.parameters.input_width)
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.parameters.data_dir,
				self.parameters.data_list,
				input_size,
				self.parameters.random_scale,
				self.parameters.random_mirror,
				self.parameters.ignore_label,
				IMG_MEAN,
				self.coord)
			self.image_batch, self.label_batch = reader.dequeue(self.parameters.batch_size)
		
		net = Deeplab_v2(self.image_batch, self.parameters.num_classes, True, self.parameters.dilated_type)
		restore_var = [v for v in tf.global_variables() if 'fc' not in v.name and 'fix_w' not in v.name]
		all_trainable = tf.trainable_variables()
		encoder_trainable = [v for v in all_trainable if 'fc' not in v.name] 
		decoder_trainable = [v for v in all_trainable if 'fc' in v.name]
		
		decoder_w_trainable = [v for v in decoder_trainable if 'weights' in v.name or 'gamma' in v.name] 
		decoder_b_trainable = [v for v in decoder_trainable if 'biases' in v.name or 'beta' in v.name] 
		assert(len(all_trainable) == len(decoder_trainable) + len(encoder_trainable))
		assert(len(decoder_trainable) == len(decoder_w_trainable) + len(decoder_b_trainable))

		raw_output = net.outputs 

		output_shape = tf.shape(raw_output)
		output_size = (output_shape[1], output_shape[2])

		label_proc = prepare_label(self.label_batch, output_size, num_classes=self.parameters.num_classes, one_hot=False)
		raw_gt = tf.reshape(label_proc, [-1,])
		indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.parameters.num_classes - 1)), 1)
		gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
		raw_prediction = tf.reshape(raw_output, [-1, self.parameters.num_classes])
		prediction = tf.gather(raw_prediction, indices)

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
		# L2 regularization
		l2_losses = [self.parameters.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
		# Loss function
		self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

		base_lr = tf.constant(self.parameters.learning_rate)
		self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
		learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.parameters.num_steps), self.parameters.power))

		opt_encoder = tf.train.MomentumOptimizer(learning_rate, self.parameters.momentum)
		opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.parameters.momentum)
		opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.parameters.momentum)

		grads = tf.gradients(self.reduced_loss, encoder_trainable + decoder_w_trainable + decoder_b_trainable)
		grads_encoder = grads[:len(encoder_trainable)]
		grads_decoder_w = grads[len(encoder_trainable) : (len(encoder_trainable) + len(decoder_w_trainable))]
		grads_decoder_b = grads[(len(encoder_trainable) + len(decoder_w_trainable)):]

		train_op_conv = opt_encoder.apply_gradients(zip(grads_encoder, encoder_trainable))
		train_op_fc_w = opt_decoder_w.apply_gradients(zip(grads_decoder_w, decoder_w_trainable))
		train_op_fc_b = opt_decoder_b.apply_gradients(zip(grads_decoder_b, decoder_b_trainable))

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance
		with tf.control_dependencies(update_ops):
			self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

		self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)


		self.loader = tf.train.Saver(var_list=restore_var)


		raw_output_up = tf.image.resize_bilinear(raw_output, input_size)
		raw_output_up = tf.argmax(raw_output_up, axis=3)
		self.pred = tf.expand_dims(raw_output_up, dim=3)
		# Image summary.
		images_summary = tf.py_func(inv_preprocess, [self.image_batch, 2, IMG_MEAN], tf.uint8)
		labels_summary = tf.py_func(decode_labels, [self.label_batch, 2, self.parameters.num_classes], tf.uint8)
		preds_summary = tf.py_func(decode_labels, [self.pred, 2, self.parameters.num_classes], tf.uint8)
		self.total_summary = tf.summary.image('images',
			tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
			max_outputs=2) # Concatenate row-wise.
		if not os.path.exists(self.parameters.logdir):
			os.makedirs(self.parameters.logdir)
		self.summary_writer = tf.summary.FileWriter(self.parameters.logdir, graph=tf.get_default_graph())

	def test_setup(self):
		self.coord = tf.train.Coordinator()

		with tf.name_scope("create_inputs"):
			reader = ImageReader(self.parameters.data_dir,self.parameters.valid_data_list,None, False, False, self.parameters.ignore_label,IMG_MEAN,self.coord)
			image, label = reader.image, reader.label 
		self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
		
		net = Deeplab_v2(self.image_batch, self.parameters.num_classes, False, self.parameters.dilated_type)


		raw_output = net.outputs
		raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3,])
		raw_output = tf.argmax(raw_output, axis=3)
		pred = tf.expand_dims(raw_output, dim=3)
		self.pred = tf.reshape(pred, [-1,])

		gt = tf.reshape(self.label_batch, [-1,])
		temp = tf.less_equal(gt, self.parameters.num_classes - 1)
		weights = tf.cast(temp, tf.int32)

		gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

		self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
			self.pred, gt, weights=weights)

		self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
			self.pred, gt, num_classes=self.parameters.num_classes, weights=weights)

		self.parametersusion_matrix = tf.contrib.metrics.confusion_matrix(
			self.pred, gt, num_classes=self.parameters.num_classes, weights=weights)
		self.loader = tf.train.Saver(var_list=tf.global_variables())

	def predict_setup(self):
		self.coord = tf.train.Coordinator()
		with tf.name_scope("create_inputs"):
			reader = ImageReader(self.parameters.data_dir,self.parameters.test_data_list,None,False,False,self.parameters.ignore_label,IMG_MEAN,self.coord)
			image, label = reader.image, reader.label 
		image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)

		net = Deeplab_v2(image_batch, self.parameters.num_classes, False, self.parameters.dilated_type)
		raw_output = net.outputs
		raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
		raw_output = tf.argmax(raw_output, axis=3)
		self.pred = tf.cast(tf.expand_dims(raw_output, dim=3), tf.uint8)

		if not os.path.exists(self.parameters.out_dir):
			os.makedirs(self.parameters.out_dir)
			os.makedirs(self.parameters.out_dir + '/prediction')
			if self.parameters.visual:
				os.makedirs(self.parameters.out_dir + '/visual_prediction')

		self.loader = tf.train.Saver(var_list=tf.global_variables())

	def save(self, saver, step):

		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.parameters.modeldir, model_name)
		if not os.path.exists(self.parameters.modeldir):
			os.makedirs(self.parameters.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')

	def load(self, saver, filename):
		saver.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))

	def compute_IoU_per_class(self, confusion_matrix):
		mIoU = 0
		for i in range(self.parameters.num_classes):
			# IoU = true_positive / (true_positive + false_positive + false_negative)
			TP = confusion_matrix[i,i]
			FP = np.sum(confusion_matrix[:, i]) - TP
			FN = np.sum(confusion_matrix[i]) - TP
			IoU = TP / (TP + FP + FN)
			print ('class %d: %.3f' % (i, IoU))
			mIoU += IoU / self.parameters.num_classes
		print ('mIoU: %.3f' % mIoU)
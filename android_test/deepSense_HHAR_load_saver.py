import tensorflow as tf 
import numpy as np

import plot

import time
import math
import os
import sys

layers = tf.contrib.layers 

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

# BATCH_SIZE = 64
BATCH_SIZE = 1
TOTAL_ITER_NUM = 1000000000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))

###### Import training data
def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	defaultVal = [[0.] for idx in range(WIDE*FEATURE_DIM + OUT_DIM)]

	fileData = tf.decode_csv(value, record_defaults=defaultVal)
	features = fileData[:WIDE*FEATURE_DIM]
	features = tf.reshape(features, [WIDE, FEATURE_DIM])
	labels = fileData[WIDE*FEATURE_DIM:]
	return features, labels

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	# filename_queue = tf.train.string_input_producer(filenames, num_epochs=TOTAL_ITER_NUM*EVAL_ITER_NUM*10000000, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	min_after_dequeue = 1000#int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	return example_batch, label_batch

######

# def batch_norm_layer(inputs, phase_train, scope=None):
# 	return tf.cond(phase_train,  
# 		lambda: layers.batch_norm(inputs, is_training=True, scale=True, 
# 			updates_collections=None, scope=scope),  
# 		lambda: layers.batch_norm(inputs, is_training=False, scale=True,
# 			updates_collections=None, scope=scope, reuse = True)) 

def batch_norm_layer(inputs, phase_train, resuse=False, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True, 
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = resuse)
	# else:
	# 	return layers.batch_norm(inputs, is_training=False, scale=True,
	# 		updates_collections=None, scope=scope, reuse = True)

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)

		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

		acc_conv1 = layers.convolution2d(acc_inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
						stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv1')
		acc_conv1 = batch_norm_layer(acc_conv1, train, scope='acc_BN1')
		acc_conv1 = tf.nn.relu(acc_conv1)
		acc_conv1_shape = acc_conv1.get_shape().as_list()
		acc_conv1 = layers.dropout(acc_conv1, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[acc_conv1_shape[0], 1, 1, acc_conv1_shape[3]], scope='acc_dropout1')

		acc_conv2 = layers.convolution2d(acc_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv2')
		acc_conv2 = batch_norm_layer(acc_conv2, train, scope='acc_BN2')
		acc_conv2 = tf.nn.relu(acc_conv2)
		acc_conv2_shape = acc_conv2.get_shape().as_list()
		acc_conv2 = layers.dropout(acc_conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[acc_conv2_shape[0], 1, 1, acc_conv2_shape[3]], scope='acc_dropout2')

		acc_conv3 = layers.convolution2d(acc_conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv3')
		acc_conv3 = batch_norm_layer(acc_conv3, train, scope='acc_BN3')
		acc_conv3 = tf.nn.relu(acc_conv3)
		acc_conv3_shape = acc_conv3.get_shape().as_list()
		acc_conv_out = tf.reshape(acc_conv3, [acc_conv3_shape[0], acc_conv3_shape[1], 1, acc_conv3_shape[2],acc_conv3_shape[3]])


		gyro_conv1 = layers.convolution2d(gyro_inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
						stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope='gyro_conv1')
		gyro_conv1 = batch_norm_layer(gyro_conv1, train, scope='gyro_BN1')
		gyro_conv1 = tf.nn.relu(gyro_conv1)
		gyro_conv1_shape = gyro_conv1.get_shape().as_list()
		gyro_conv1 = layers.dropout(gyro_conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[gyro_conv1_shape[0], 1, 1, gyro_conv1_shape[3]], scope='gyro_dropout1')

		gyro_conv2 = layers.convolution2d(gyro_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='gyro_conv2')
		gyro_conv2 = batch_norm_layer(gyro_conv2, train, scope='gyro_BN2')
		gyro_conv2 = tf.nn.relu(gyro_conv2)
		gyro_conv2_shape = gyro_conv2.get_shape().as_list()
		gyro_conv2 = layers.dropout(gyro_conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[gyro_conv2_shape[0], 1, 1, gyro_conv2_shape[3]], scope='gyro_dropout2')

		gyro_conv3 = layers.convolution2d(gyro_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', data_format='NHWC', scope='gyro_conv3')
		gyro_conv3 = batch_norm_layer(gyro_conv3, train, scope='gyro_BN3')
		gyro_conv3 = tf.nn.relu(gyro_conv3)
		gyro_conv3_shape = gyro_conv3.get_shape().as_list()
		gyro_conv_out = tf.reshape(gyro_conv3, [gyro_conv3_shape[0], gyro_conv3_shape[1], 1, gyro_conv3_shape[2], gyro_conv3_shape[3]])	


		sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()	
		sensor_conv_in = layers.dropout(sensor_conv_in, CONV_KEEP_PROB, is_training=train,
			noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], scope='sensor_dropout_in')

		sensor_conv1 = layers.convolution2d(sensor_conv_in, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv1')
		sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
		sensor_conv1 = tf.nn.relu(sensor_conv1)
		sensor_conv1_shape = sensor_conv1.get_shape().as_list()
		sensor_conv1 = layers.dropout(sensor_conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[sensor_conv1_shape[0], 1, 1, 1, sensor_conv1_shape[4]], scope='sensor_dropout1')

		sensor_conv2 = layers.convolution2d(sensor_conv1, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN2],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv2')
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
		sensor_conv2 = tf.nn.relu(sensor_conv2)
		sensor_conv2_shape = sensor_conv2.get_shape().as_list()
		sensor_conv2 = layers.dropout(sensor_conv2, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[sensor_conv2_shape[0], 1, 1, 1, sensor_conv2_shape[4]], scope='sensor_dropout2')

		sensor_conv3 = layers.convolution2d(sensor_conv2, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN3],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv3')
		sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
		sensor_conv3 = tf.nn.relu(sensor_conv3)
		sensor_conv3_shape = sensor_conv3.get_shape().as_list()
		sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]*sensor_conv3_shape[4]])

		gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
		# if train:
		# 	gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

		gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
		# if train:
		# 	gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

		cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
		init_state = cell.zero_state(BATCH_SIZE, tf.float32)

		cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)

		sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
		avg_cell_out = sum_cell_out/avgNum

		logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		return logits



batch_eval_feature = tf.placeholder(tf.float32, shape=[BATCH_SIZE, WIDE*FEATURE_DIM], name='I')
batch_eval_feature = tf.reshape(batch_eval_feature, [BATCH_SIZE, WIDE, FEATURE_DIM])
batch_eval_label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUT_DIM], name='L')

logits_eval_org = deepSense(batch_eval_feature, False, name='deepSense')
logits_eval = tf.identity(logits_eval_org, name='O')
predict_eval = tf.argmax(logits_eval, axis=1)
loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))



saver = tf.train.Saver()

LOAD_DIR = 'model_saver' # your own pre-trained model dir
LOAD_FILE = 'model.ckpt-2149' #your own pre-trained model file

ANDROID_SAVE_DIR = 'android_model_saver'
if not os.path.exists(ANDROID_SAVE_DIR):
	os.mkdir(ANDROID_SAVE_DIR)

with tf.Session() as sess:


	saver.restore(sess, os.path.join(LOAD_DIR, LOAD_FILE))

	# save the graph
	tf.train.write_graph(sess.graph_def, ANDROID_SAVE_DIR, 'tfdroid.pbtxt')  
	#save a checkpoint file, which will store the above assignment 
	saver.save(sess, os.path.join(ANDROID_SAVE_DIR, 'tfdroid.ckpt'))







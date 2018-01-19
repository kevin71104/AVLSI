################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
import argparse
from numpy import random

import tensorflow as tf
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

#### Construct the argument parse and parse the arguments ####
ap = argparse.ArgumentParser()
ap.add_argument("-util", "--util", type=str, default="acc",
        help="choose the type of operation: accuracy or other features "
              "including FLOPS, Inference time, and # of parameters")

args = vars(ap.parse_args())

if args["util"] not in ("acc", "others"):
    raise AssertionError("The --util command line argument should "
		"be acc or others")

################################################################################
# LOAD Labels
label = np.loadtxt("label/val_id.txt", dtype=str)
num_image = label.shape[0]
top1 = 0.0
top5 = 0.0
error1 = ''
error5 = ''
#count_time = 0.0  # cpu time

#num_image=100

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0], name='conv1_W')
conv1b = tf.Variable(net_data["conv1"][1], name='conv1_b')
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name='norm1')

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name='maxpool_1')


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0],name='conv2_W')
conv2b = tf.Variable(net_data["conv2"][1],name='conv2_b')
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name='norm2')

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name='maxpool_2')

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0],name='conv3_W')
conv3b = tf.Variable(net_data["conv3"][1],name='conv3_b')
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0],name='conv4_W')
conv4b = tf.Variable(net_data["conv4"][1],name='conv4_b')
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0],name='conv5_W')
conv5b = tf.Variable(net_data["conv5"][1],name='conv5_b')
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name='maxpool_5')

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0],name='fc6_W')
fc6b = tf.Variable(net_data["fc6"][1],name='fc6_b')
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0],name='fc7_W')
fc7b = tf.Variable(net_data["fc7"][1],name='fc7_b')
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0],name='fc8_W')
fc8b = tf.Variable(net_data["fc8"][1],name='fc8_b')
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
output_prediction = tf.nn.softmax(fc8)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

################################################################################
#Read Image, and change to BGR
inputShape = (227, 227)
preprocess = imagenet_utils.preprocess_input
if args["util"] == "acc":

    for k in range(1,num_image+1):
        path = "ILSVRC2012_img_val/ILSVRC2012_val_000"
        if k < 10:
            pic_name = '0000'+str(k)
        elif 10 <= k and k < 100:
            pic_name = '000'+str(k)
        elif 100 <= k and k < 1000:
            pic_name = '00'+str(k)
        elif 1000 <= k and k < 10000:
            pic_name = '0' +str(k)
        else:
            pic_name = str(k)

        pic_path = path + pic_name + ".JPEG"
        image = load_img(pic_path, target_size=inputShape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess(image)
        if (k%10 == 0):
            print("Progress: {}/50000".format(k),"top1 acc:", round(top1/k,4),"top5 acc:", round(top5/k,4))

        #Output:
        output = sess.run(output_prediction, feed_dict = {x:[image[0,:,:,:]]})

        P = imagenet_utils.decode_predictions(output)
        if label[k-1] in P[0][0]:
            top1 += 1
        else:
            error1 += str(k) + '\n'

        inTop5 = False
        for (i, (imagenetID, pred_label, prob)) in enumerate(P[0]):
            if label[k-1] == imagenetID:
                top5 += 1
                inTop5 = True

        if inTop5 == False:
            error5 += str(k) + '\n'

    print('total image', num_image)
    print('top_1',top1/num_image)
    print('top_5',top5/num_image)

    of1 = open("./record/alexnet_error1.txt",'w')
    of2 = open("./record/alexnet_error5.txt",'w')
    of1.write(error1)
    of1.close()
    of2.write(error5)
    of2.close()
else:
##### FLOAT_OPS_COUNT & Parameters #####
    image = load_img("elephant.png", target_size=inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess(image)

    run_meta = tf.RunMetadata()
    output = sess.run(output_prediction, feed_dict = {x:[image[0,:,:,:]]},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata = run_meta)
    flops=tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_cmd='graph',
            run_meta=run_meta,
            tfprof_options=tf.profiler.ProfileOptionBuilder.float_operation())
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_cmd='scope',
            tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

    op_time = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            run_meta=run_meta,
            tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

    print("Total FLOPS:",flops.total_float_ops)
    print("Total Parameters:", param_stats.total_parameters)
    print("Total Execution Time: {} ms".format(round(op_time.total_cpu_exec_micros/1000,2)))



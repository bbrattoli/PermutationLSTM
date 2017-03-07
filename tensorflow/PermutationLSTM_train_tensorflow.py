# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:41:33 2017

@author: Biagio Brattoli
"""

import os
import numpy as np
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from random import shuffle
import argparse

from LoadSequencesUtils import *

import alexnet_LSTM
import tensorflow as tf

def max_lenght(img_list):
    F = 0
    for frames in img_list:
        F = max(F,len(frames))
    return F

def next_train_batch(images_dict, img_list, F):
    N = len(img_list)
    
    batch = np.zeros((2*F*N,227,227,3),dtype=np.float32)
    labels = np.empty((2*N,),dtype=np.int)
    seq_len = np.zeros(N*2,dtype=np.int)
    
    for n in range(N):
        imgSeq_pos = np.empty((F,227,227,3),dtype=np.float32)
        frames = img_list[n]
        
        augParams = getAugParameters()
        for f in range(len(frames)):
            image = images_dict[frames[f]]
            image = augmentFrame(image,augParams)
            
            if image.shape[0]!=227:
                im = np.array(Image.fromarray(image).resize((227,227),Image.BILINEAR))
            else:
                im = image
            
            imgSeq_pos[f,:,:,:] = im.astype(np.float32)
        
        imgSeq_neg = imgSeq_pos[np.random.permutation(F),:,:,:]
        
        batch[n*2*F:(n+1)*2*F,:,:,:] = np.concatenate((imgSeq_pos,imgSeq_neg),axis=0)
        labels[n*2] = 0
        labels[n*2+1] = 1
        seq_len[n*2] = len(frames)
        seq_len[n*2+1] = len(frames)
    
    return {'data':batch,'labels':labels,'seq_len':seq_len}

def init_network(weights,batch_size=30,LR=1e-3,F=10):
    net = alexnet_LSTM.Alexnet_LSTM(num_classes=2, num_layers_to_init=7, \
                    gpu_memory_fraction=1.0,max_length=F,num_hidden=1024, \
                    batch_size = batch_size, train_conv=True)
    
    l_pred = tf.argmax(net.prob, dimension=1)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.logits,labels=net.l_true)
    cost = tf.reduce_mean(cross_entropy)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
    #optimizer = tf.train.AdamOptimizer(1e-5)
    optimizer = tf.train.AdagradOptimizer(learning_rate=LR)
    
    train_op = optimizer.minimize(cost,global_step = net.global_iter_counter)
    
    correct_prediction = tf.equal(l_pred,tf.cast(net.l_true,tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    tf.summary.scalar('loss',cost)
    tf.summary.scalar('acc',accuracy)
    
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(net.x, [-1, 227, 227, 3])
        tf.summary.image('input', image_shaped_input, 5)
    
    with tf.name_scope('conv1_filters'):
        filter_image = tf.transpose(net.trainable_vars['conv1w'],[3,0,1,2])
        tf.summary.image('filters',filter_image,90)
    
    net.sess.run(tf.global_variables_initializer())
    if weights is not None:
        net.restore_from_snapshot(weights,7)
    
    return net, train_op, accuracy, cost

###########################  INITIALIZE NETWORK  #################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Permutation LSTM loading sequences from a txt file.')
    parser.add_argument('sequences_file', metavar='input.txt', type=str, nargs='?',
                        help='the txt file containing the sequences')
    parser.add_argument('images_path', type=str, nargs='?',
                        help='path to concatenate to the images path from the sequence_file')
    
    parser.add_argument('--learning_rate', type=float, nargs='?',help='learning rate')
    parser.add_argument('--max_iterations', type=int, nargs='?',help='max iterations')
    parser.add_argument('--batch_size', type=int, nargs='?',help='half batch size')
    
    parser.add_argument('--outpath', type=str, nargs='?',help='location for saving the log')
    parser.add_argument('--weights', type=str, nargs='?',help='initialize weights')
#    parser.add_argument('--weights', metavar='', type=open, nargs='?',help='initialize weights')
    args = parser.parse_args()
    
    images_list_txt = args.sequences_file
    img_path = args.images_path
    out_path = args.outpath
    weights = args.weights
    LR = args.learning_rate
    max_iterations = args.max_iterations
    BS = args.batch_size
    
    if max_iterations is None:
        max_iterations = 30000
    
    if LR is None:
        LR = 1e-3
    
    if BS is None:
        BS = 30
    
    if out_path is None:
        out_path = '.'
    
    summary_step = 10
    saver_step = 500
    
    ###################### LOAD DATA ################
    print 'Reading sequences'
    t = time()
    img_list = load_list(images_list_txt)
    F = max_lenght(img_list)
    print 'Longest sequence: %d'%F
    
    images_dict = load_data(img_path, img_list)
    shuffle(img_list)
    print 'Data loaded in %.2f seconds'%(time()-t)
    
    ###################### NETWORK ################
    print 'Initializing the network'
    net, train_op, accuracy, cost = init_network(weights,batch_size=batch_size*2,LR=LR,F=F)
    print 'Network initialized'
    
    ###################### TRAIN ################
    N = len(img_list)
    
    summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(out_path+'/train',net.sess.graph)
    test_summary_writer  = tf.summary.FileWriter(out_path+'/test')
    
    B = np.floor(N/BS).astype('int')
    
    saver = tf.train.Saver(max_to_keep=10000)
    
    print 'Start training'
    for ii in xrange(max_iterations):
        t1 = time()
        batch_idx = ii%B
        
        batch = next_train_batch(images_dict,img_list[batch_idx*BS:(batch_idx+1)*BS],F)
        
        feed_dict_train = {net.x: batch['data'],net.l_true: batch['labels'],
                           net.is_phase_train: 1, net.seq_len: batch['seq_len']}
        
        if ii%summary_step==0:
            _,summary_str = net.sess.run([train_op,summary_op],feed_dict=feed_dict_train)
            train_summary_writer.add_summary(summary_str,ii)
        else:
            _,acc,loss = net.sess.run([train_op,accuracy,cost], feed_dict=feed_dict_train)
            print 'Iter %d, Loss=%.3f, Train Accuracy= %.3f done in %.3f'%(ii,loss,acc,time()-t1)
        
        if ii%saver_step == 0 and ii>0:
            saver.save(net.sess,out_path,global_step = ii)
            print 'Checkpoint '+out_path





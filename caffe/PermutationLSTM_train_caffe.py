# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:41:33 2017

@author: Biagio Brattoli
"""

import os
import numpy as np
from time import time
from PIL import Image
from random import shuffle
import argparse

from LoadSequencesUtils import *

import caffe

def max_lenght(img_list):
    F = 0
    for frames in img_list:
        F = max(F,len(frames))
    return F

def next_train_batch(images_dict, img_list, F):
    N = len(img_list)
    
    batch = np.zeros((2*F*N,227,227,3),dtype=np.float32)
    labels = np.empty((2*N,),dtype=np.int)
    
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
    
    cm = np.ones((L,2*N))
    cm[0,:]=0
    
    batch = batch.transpose((0,3, 1, 2))
    return {'data':batch,'labels':labels,'clip_mark':cm}


def write_solver(solver_path,net_name,LR=0.001,snapshot_path='snapshots',GPU=0,epoch_iter=-1,test_iter=-1):
#    test_iter=np.floor(Ntest/3)
#    epoch_iter=np.floor(Ntrain/24)
    f = open(solver_path,"w")
    f.write('net: "'+ net_name +'.prototxt"\n')
    f.write('test_iter: %d\n'%test_iter)
    f.write('test_interval: 1000000000\n')
    f.write('base_lr: %f\n'%LR)
    f.write('lr_policy: "step"\n')
    f.write('gamma: 0.1\n')
    f.write('stepsize: %d\n'%27000)#(epoch_iter*30))
    f.write('display: %d\n'%10)
    f.write('max_iter: 1000000\n')
    f.write('momentum: 0.9\n')
    f.write('weight_decay: 0.005\n')
    f.write('snapshot: %d\n'%10000)
    f.write('snapshot_prefix: "'+snapshot_path+'"\n')
    f.write('solver_mode: GPU\n')
    f.write('device_id: %d\n'%GPU)
    f.close()

RED   = '\033[91m'
END   = '\033[0m'

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
    parser.add_argument('--gpu_id', type=int, nargs='?',help='half batch size')
    
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
    gpu_id = args.gpu_id
    
    if max_iterations is None:
        max_iterations = 30000
    
    if LR is None:
        LR = 1e-3
    
    if BS is None:
        BS = 40
    
    if gpu_id is None:
        gpu_id = 0
    
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
    net_path       = './network/'
    solver_path    = net_path+'solver.prototxt'
    net_name       = net_path+'train_test_lstm'
    
    snapshot_name = out_path+'PermutationLstm'
    if not os.path.exists(snapshot_path):
        os.mkdir(snapshot_path)
    
    write_solver(solver_path,net_name,LR,snapshot_name,gpu_id,B,B2)
    
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_path)
    
    net = solver.net
    solver.net.copy_from(weights)
    
    print 'Network initialized'
    
    ###################### TRAIN ################
    N = len(img_list)
    
    text_file = open('permutaiton%d_10frames_train_lr%.3f.log'%(data.L,LR), 'w')
    try:
        for ii in xrange(max_iterations):
            t1 = time()
            epoch = ii/B
            batch_idx = ii%B
            
            batch = next_train_batch(images_dict,img_list[batch_idx*BS:(batch_idx+1)*BS],F)
            
            net.blobs['data'].data[...]  = batch['data']
            net.blobs['label'].data[...] = batch['labels']
            net.blobs['cm'].data[...]    = batch['clip_mark']
            solver.step(1)
            
            loss = net.blobs['loss'].data
            acc  = net.blobs['accuracy'].data
            
            print RED+'Epoch %d[%d]: Accuracy %.3f Loss %.3f'%(epoch,ii,acc,loss)+END
            text_file.write('Epoch %d[%d]: Accuracy %.3f Loss %.3f\n'%(epoch,ii,acc,loss))
    
    except KeyboardInterrupt:
        text_file.close()
    
    text_file.close()





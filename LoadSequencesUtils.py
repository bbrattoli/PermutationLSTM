# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 08:55:49 2017

@author: bbrattol
"""
import os
import sys
from time import time
from tqdm import trange

import numpy as np
from PIL import Image
from PIL import ImageEnhance

from joblib import Parallel, delayed

HEIGHT = 227
WIDTH = 227

def load_list(images_list_txt):
    img_list = []
    with open(images_list_txt,'r') as f:
        row = f.readlines()
        for r in range(len(row)):
            img_list.append(row[r].split())
    
    return img_list

def collect_frame_names(img_list):
    images_dict = dict()
    for i in range(len(img_list)):
        frames = img_list[i]
        for t in range(len(frames)):
            images_dict[frames[t]] = np.zeros([HEIGHT,WIDTH,3],'uint8')
    
    return images_dict

def single_frame(img_path,file_name):
    image = np.array(Image.open(img_path+file_name+'.jpg'), dtype=np.uint8)    
    if len(image.shape)==2:
        image = np.repeat(image[:,:,np.newaxis],3,axis=2)
    return image

def load_data(img_path, img_list, parallel=True):
    images_dict = collect_frame_names(img_list)
    image_names = images_dict.keys()
    
    t1 = time()
    if parallel:
        with Parallel(n_jobs=32) as parallel:
            images_list = parallel(delayed(single_frame)(img_path,name) for name in image_names)
    else:
        images_list = []
        for i in trange(len(image_names)):
            img = single_frame(img_path,image_names[i])
            images_list.append(img)
    
    print '%d images loaded in %.2f'%(len(images_list),time()-t1)
    
    for i in range(len(image_names)):
        images_dict[image_names[i]] = images_list[i]
    
    return images_dict

def getAugParameters():
    
    augParams = dict([(key, []) for key in {"scaling","translation","contrast","brightness"}])
    augParams['scaling'] = np.random.randint(45,55)
    augParams['translation'] = np.random.randint(-10,10,2)
    augParams['contrast'] = np.random.uniform(low=0.7, high=1.5,size = (1))
    augParams['brightness'] = np.random.uniform(low=0.7, high=1.5,size = (1))
    
    return augParams

def augmentFrame(img,augParams):
    coord_orig = [HEIGHT/2,WIDTH/2]
    box = augParams['scaling']
    coord = coord_orig+augParams['translation']
    img = img[coord[1]-box:coord[1]+box,coord[0]-box:coord[0]+box,:].copy()
    
    img = Image.fromarray(img)
    bright = ImageEnhance.Brightness(img)
    img = bright.enhance(augParams['brightness'])
    contr = ImageEnhance.Contrast(img)
    img = contr.enhance(augParams['contrast'])
    img = np.array(img, dtype=np.uint8)
    
    return img

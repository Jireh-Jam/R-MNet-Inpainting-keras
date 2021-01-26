# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:58:05 2021

@author: 18060721
"""
import os
import cv2
import numpy as np
from copy import deepcopy
from rmnet import RMNETWGAN
from config import config



#Config Loader
CONFIG_FILE = './config/config.ini'
config = config.MainConfig(CONFIG_FILE).testing

#Data params
test_img_dir ='./images/celebA_HQ_test/'
test_mask_dir ='./masks/test_masks/'
test_imgs_path = os.listdir(test_img_dir)
test_masks_path = os.listdir(test_mask_dir) 

#Directories
imgs_dir = 'real_images_rmnet'
masked_dir = 'masked_images_rmnet'
inpainted_dir = 'inpainted_images_rmnet'
trained_model_path = r'./models/RMNet_WACV2021'

#Load data
def generate_test_batch(last_img_on, imgs_in_batch):  
    
    if (last_img_on + imgs_in_batch) >= len(test_imgs_path): 
        imgs_in_batch = len(test_imgs_path)-last_img_on 
    imgs = np.zeros((config.imgs_in_batch,config.img_width,config.img_height,config.channels))
    masked_imgs = np.zeros((config.imgs_in_batch,config.img_width,config.img_height,config.channels))
    masks = np.zeros((config.imgs_in_batch,config.img_width,config.img_height,config.mask_channels))
    idx = 0
    for i in range(imgs_in_batch):  
        print("\rloading Image " + str(i) + ' of ' +str(len(test_imgs_path)), end=" ") 
        img = (cv2.imread(test_img_dir+test_imgs_path[last_img_on],1))
        img = cv2.resize(img,(config.img_width, config.img_height))
        img = img[..., [2, 1, 0]]
        img = (img - 127.5) / 127.5
        mask = (cv2.imread(test_mask_dir+test_masks_path[last_img_on],0))
        mask[mask == 255] = 1
        mask = cv2.resize(mask,(config.img_width, config.img_height))        
        mask = np.reshape(mask,(config.img_width,config.img_height,config.mask_channels))
        masks[i] = mask
        masked_imgs[i] = deepcopy(img)
        masked_imgs[i][np.where((mask == [1,1,1]).all(axis=2))]=[255,255,255]
        imgs[i] = img 
        last_img_on += 1
        idx+=1               
    return last_img_on, imgs,masks,masked_imgs 

#Inpaint imgaes 
def inpaint():
    imgs_in_batch = config.imgs_in_batch
    last_img_on =config.last_img_on
    rmnet_model = RMNETWGAN(config)
    #Edit last_trained_epoch in config.ini
    rmnet_model.generator.load_weights('{}/weight_{}.h5'.format(trained_model_path,config.last_trained_epoch))
    for i in range(len(test_imgs_path)):
        if not os.path.exists(inpainted_dir):
            os.makedirs(inpainted_dir)
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)   
        if not os.path.exists(masked_dir):
            os.makedirs(masked_dir)              
        last_img_on, imgs, masks,masked_imgs  = generate_test_batch(last_img_on, imgs_in_batch)   
        gen_imgs = rmnet_model.generator.predict([imgs,masks], config.batch_size) 
        gen_imgsRGB = gen_imgs[:,:,:,0:3]
        input_img = np.expand_dims(imgs[0], 0)
        input_mask = np.expand_dims(masks[0], 0)            
        maskedImg = ((1 - input_mask)*input_img) + input_mask        
        cv2.imwrite(r'./' + imgs_dir +'/'+str(i) +'.jpg',(imgs[0][..., [2, 1, 0]]* 127.5 + 127.5).astype("uint8"))
        cv2.imwrite(r'./' + masked_dir +'/' +str(i) +'.jpg',(maskedImg[0][..., [2, 1, 0]]* 127.5 + 127.5).astype("uint8"))
        cv2.imwrite(r'./' + inpainted_dir +'/' +str(i) +'.jpg',(gen_imgsRGB[0][..., [2, 1, 0]]* 127.5 + 127.5).astype("uint8"))
        
if __name__=='__main__':
    inpaint()
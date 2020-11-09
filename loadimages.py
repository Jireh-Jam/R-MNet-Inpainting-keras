# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:05:03 2020

@author: 18060721
"""

import os
import numpy as np
import cv2
import random
import datetime



imgsInDir = r'./CelebA-HQ-train'
maskInDir = r'./masks'
imgsInPath = os.listdir(imgsInDir)
maskInPath = os.listdir(maskInDir)
img_width = 256
img_height = 256
def GenerateTrainingBatch(lastImageOn,imagesInBatch):
    if(lastImageOn + imagesInBatch) >= len(imgsInPath):
        imagesInBatch = len(imgsInPath)- lastImageOn
    imgs = np.zeros((imagesInBatch,img_width,img_height,3))
    masks = np.zeros((imagesInBatch,img_width,img_height,1))
    masks_idx = random.sample(range(1,len(maskInPath)), imagesInBatch)
    idx = 0
    for i in range(imagesInBatch):
        print("\rLoading Image " +(datetime.datetime.now().strftime("%c")+" "+str(i)), end=" ")
        img = cv2.imread(os.path.join(imgsInDir,imgsInPath[lastImageOn])) 
        mask = cv2.imread(os.path.join(maskInDir,maskInPath[masks_idx[idx]]),0)
        #mask = 1-mask # Uncomment if using Quick-draw mask dataset
        mask = np.reshape(mask,(img_width,img_height,1))
        mask = np.expand_dims(mask, axis=-1)    
        masks[i] = mask
        imgs[i] = img       
        lastImageOn +=1
        idx+=1
        if(lastImageOn >= len(imgsInPath)):
            lastImageOn = 0    
#    imgs = (imgs).astype("float32")/255 
#    masks = (masks).astype("float32")/255
#    #masks = 1-masks
#    cv2.imshow("img",((imgs[0])*255).astype("uint8")) 
#    cv2.imshow("mask",((masks[0])*255).astype("uint8"))
#    cv2.imshow("muliply",(((imgs[0]*masks[0])*255).astype("uint8")))
#    cv2.waitKey(0)
    return lastImageOn, imgs, masks

import os
import cv2
import numpy as np
from copy import deepcopy
from rmnet import RMNETWGAN
from config import config    
import random
    
# =================================================================================== #
#               1. Config Loader                                                      #
# =================================================================================== #  

CONFIG_FILE = './config/config.ini'
config = config.MainConfig(CONFIG_FILE).testing

# =================================================================================== #
#               2. Data params                                                        #
# =================================================================================== # 

test_img_dir ='./images/celebA_HQ_test/'
test_mask_dir ='./masks/test_masks/'
test_imgs_path = os.listdir(test_img_dir)
test_masks_path = os.listdir(test_mask_dir) 

# =================================================================================== #
#               3. Directories                                                        #
# =================================================================================== # 

imgs_dir = 'real_images_rmnet'
masked_dir = 'masked_images_rmnet'
inpainted_dir = 'inpainted_images_rmnet'
trained_model_path = r'./models/RMNet_WACV2021'  

# =================================================================================== #
#               4. Data Loader                                                        #
# =================================================================================== # 

def generate_test_batch(last_img_on, imgs_in_batch):  
    
    if (last_img_on + imgs_in_batch) >= len(test_imgs_path): 
        imgs_in_batch = len(test_imgs_path)-last_img_on 
    imgs = np.zeros((config.imgs_in_batch,config.img_width,config.img_height,config.channels))
    masked_imgs = np.zeros((config.imgs_in_batch,config.img_width,config.img_height,config.channels))
    masks = np.zeros((config.imgs_in_batch,config.img_width,config.img_height,config.mask_channels))
    for i in range(imgs_in_batch):
        print("\rLoading image number "+ str(i) + " of " + str(len(test_imgs_path)), end = " ")
        img = cv2.imread(test_img_dir+test_imgs_path[last_img_on],1).astype('float')/ 127.5 -1
        img = cv2.resize(img,(config.img_width, config.img_height))
        #If Mask regions are white, DO NOT subtract from 1. 
        #If mask regions are black, subtract from 1.
        mask = 1-cv2.imread(test_mask_dir+test_masks_path[last_img_on],0).astype('float')/ 255
        mask = cv2.resize(mask,(config.img_width, config.img_height))        
        mask = np.reshape(mask,(config.img_width,config.img_height,config.mask_channels))

        masks[i] = mask
        masked_imgs[i] = deepcopy(img)
        imgs[i] = img
        masked_imgs[i][np.where((mask == [1,1,1]).all(axis=2))]=[1,1,1]
        last_img_on += 1
        # if(last_img_on >= len(test_imgs_path)):
        #     last_img_on = 0
        # cv2.imshow("mask",((masks[0])* 255).astype("uint8"))
        # cv2.imshow("masked",((masked_imgs[0]+1)* 127.5).astype("uint8"))
        # cv2.waitKey(0 )
    return last_img_on, imgs,masks,masked_imgs    

# =================================================================================== #
#               5. Data Loader                                                        #
# =================================================================================== # 

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
    d=0    
    for i in range(3000):
        last_img_on, imgs, masks, masked_imgs = generate_test_batch(last_img_on, imgs_in_batch)            
        gen_imgs = rmnet_model.generator.predict([imgs,masks], config.batch_size) 
        gen_imgsRGB = gen_imgs[:,:,:,0:3]
        imgs = ((imgs[0]+1)* 127.5).astype("uint8")
        gen_image = ((gen_imgsRGB[0]+1)* 127.5).astype("uint8")
        mask_image = ((masked_imgs[0]+1)* 127.5).astype("uint8")
        inpainted_imgs_folder = "inpainted_images_rmnet/%d.jpg"%d
        masked_imgs_folder = "masked_images_rmnet/%d.jpg"%d
        real_imgs_folder = "real_images_rmnet/%d.jpg"%d
        cv2.imwrite(inpainted_imgs_folder,gen_image)
        cv2.imwrite(masked_imgs_folder,mask_image)
        cv2.imwrite(real_imgs_folder,imgs)
        d+=1
        
if __name__=='__main__':
    inpaint()

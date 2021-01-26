# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:35:05 2021

@author: Jireh Jam
"""

from __future__ import print_function, division


from keras.applications import VGG19
from keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Multiply, Lambda, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,MaxPooling2D,Conv2DTranspose
from keras.models import  Model
from keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model 
from keras import backend as K

import tensorflow as tf

import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import datetime
import gc

   
class RMNETWGAN():
    def __init__(self,config):
        #Input shape
        self.img_width=config.img_width
        self.img_height=config.img_height
        self.channels=config.channels
        self.mask_channles = config.mask_channels
        self.img_shape=(self.img_width, self.img_height, self.channels)
        self.img_shape_mask=(self.img_width, self.img_height, self.mask_channles)
        self.missing_shape = (self.img_width, self.img_height, self.channels)
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.sample_interval = config.sample_interval
        self.current_epoch =config.current_epoch
        self.last_trained_epoch = config.last_trained_epoch
        
        #Folders
        self.dataset_name = 'RMNet_WACV2021'
        self.models_path = 'models'
        
        #Configure Loader
        self.train_img_dir ='./images/train/'
        self.train_mask_dir ='./masks/train/'
        
        # Number of filters in the first layer of G and D
        self.gf = config.gf
        self.df = config.gf
        self.continue_train  =  True 

        
        #Optimizer
        self.g_optimizer = Adam(lr=config.g_learning_rate, 
                                beta_1=config.beta_1, 
                                beta_2=config.beta_2, 
                                epsilon=config.epsilon)
        self.d_optimizer = Adam(lr=config.d_learning_rate, 
                                beta_1=config.beta_1, 
                                beta_2=config.beta_2, 
                                epsilon=config.epsilon)
                
    # =================================================================================== #
    #                             1. Build and compile the discriminator                  #
    # =================================================================================== #
    
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[self.wasserstein_loss],
                                   optimizer=self.d_optimizer,
                                   metrics=['accuracy'])

    # =================================================================================== #
    #                             2. Build the generator                                  #
    # =================================================================================== #
        self.generator = self.build_generator()
        
        #Generator takes mask and image as input 
        image = Input(shape=self.img_shape)
        mask = Input(shape=self.img_shape_mask)
        
        #Generator predicts image
        gen_img = self.generator([image,mask])
         
        #Train the generator only for the combined model
        self.discriminator.trainable = False
         
        #Descriminator validates the predicted image
        # It takes generated images as input and determines validity
        gen_img = Lambda(lambda x : x[:,:,:,0:3])(gen_img)
        # print("this is generated image in shape {} ".format(gen_image.shape))
        valid = self.discriminator(gen_img)

    # =================================================================================== #
    #               3. The combined model (stacked generator and discriminator)           #
    #               Trains the generator to fool the discriminator                        #
    # =================================================================================== #         

        try:
            self.multi_model = multi_gpu_model(self.combined, gpus=2)
            self.multi_model.compile(loss=[self.loss, self.wasserstein_loss], loss_weights=[1.0, 1e-3], optimizer=self.g_optimizer)
            
        except:        
            self.combined = Model([image,mask], [gen_img,valid])
            self.combined.compile(loss=[self.generator_loss(mask), self.wasserstein_loss],loss_weights=[1,0.001], optimizer=self.g_optimizer)

        
    # =================================================================================== #
    #               4. Define the discriminator and generator losses                      #
    # =================================================================================== # 
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)   
       
    def generator_loss(self, mask): 
        def loss(y_true, y_pred):
            input_img = Lambda(lambda x : x[:,:,:,0:3])(y_true)
            output_img = Lambda(lambda x : x[:,:,:,0:3])(y_pred)
            reversed_mask = Lambda(self.reverse_mask,output_shape=(self.img_shape_mask))(mask)
            vgg = VGG19(include_top=False, weights='imagenet', input_shape=self.img_shape)
            loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
            loss_model.trainable = False
            perceptual_loss = (K.mean(K.square(loss_model(input_img) - loss_model(output_img))))
            masking = Multiply()([reversed_mask,input_img])
            predicting = Multiply()([reversed_mask, output_img])
            reversed_mask_loss = (K.mean(K.square(loss_model(masking) - loss_model(predicting))))
            new_loss = 0.4*perceptual_loss + 0.6*reversed_mask_loss
            return new_loss
        return loss

    # =================================================================================== #
    #               5. Define the reverese mask                                           #
    # =================================================================================== #  
        
    def reverse_mask(self,x):
        return 1-x

    # =================================================================================== #
    #               6. Define the  generator                                              #
    # =================================================================================== #  
    
    def build_generator(self):
        
        #compute inputs
        input_img = Input(shape=(self.img_shape), dtype='float32', name='image_input')
        input_mask = Input(shape=(self.img_shape_mask), dtype='float32',name='mask_input')  
        reversed_mask = Lambda(self.reverse_mask,output_shape=(self.img_shape_mask))(input_mask)
        masked_image = Multiply()([input_img,reversed_mask])
        
        #encoder
        x =(Conv2D(self.gf,(5, 5), dilation_rate=2, input_shape=self.img_shape, padding="same",name="enc_conv_1"))(masked_image)
        x =(LeakyReLU(alpha=0.2))(x)
        x =(BatchNormalization(momentum=0.8))(x)
        
        pool_1 = MaxPooling2D(pool_size=(2,2))(x) 
        
        x =(Conv2D(self.gf,(5, 5), dilation_rate=2, padding="same",name="enc_conv_2"))(pool_1)
        x =(LeakyReLU(alpha=0.2))(x)
        x =(BatchNormalization(momentum=0.8))(x)
        
        pool_2 = MaxPooling2D(pool_size=(2,2))(x) 
        
        x =(Conv2D(self.gf*2, (5, 5), dilation_rate=2, padding="same",name="enc_conv_3"))(pool_2)
        x =(LeakyReLU(alpha=0.2))(x)
        x =(BatchNormalization(momentum=0.8))(x)
        
        pool_3 = MaxPooling2D(pool_size=(2,2))(x) 
        
        x =(Conv2D(self.gf*4, (5, 5), dilation_rate=2, padding="same",name="enc_conv_4"))(pool_3)
        x =(LeakyReLU(alpha=0.2))(x)
        x =(BatchNormalization(momentum=0.8))(x)
        
        pool_4 = MaxPooling2D(pool_size=(2,2))(x) 
        
        x =(Conv2D(self.gf*8, (5, 5), dilation_rate=2, padding="same",name="enc_conv_5"))(pool_4)
        x =(LeakyReLU(alpha=0.2))(x)
        x =(Dropout(0.5))(x)
       
        #Decoder
        x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
        x =(Conv2DTranspose(self.gf*8, (3, 3), padding="same",name="upsample_conv_1"))(x)
        x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
        x =(Activation('relu'))(x)
        x =(BatchNormalization(momentum=0.8))(x)
        
        x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
        x = (Conv2DTranspose(self.gf*4, (3, 3),  padding="same",name="upsample_conv_2"))(x)
        x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
        x =(Activation('relu'))(x)
        x =(BatchNormalization(momentum=0.8))(x)
         
        x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
        x = (Conv2DTranspose(self.gf*2, (3, 3),  padding="same",name="upsample_conv_3"))(x)
        x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
        x =(Activation('relu'))(x)
        x =(BatchNormalization(momentum=0.8))(x)
         
        x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
        x = (Conv2DTranspose(self.gf, (3, 3),  padding="same",name="upsample_conv_4"))(x)
        x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
        x =(Activation('relu'))(x)
        x =(BatchNormalization(momentum=0.8))(x)
        
        x = (Conv2DTranspose(self.channels, (3, 3),  padding="same",name="final_output"))(x)
        x =(Activation('tanh'))(x)
        decoded_output = x
        reversed_mask_image = Multiply()([decoded_output, input_mask])
        output_img = Add()([masked_image,reversed_mask_image])
        concat_output_img = Concatenate()([output_img,input_mask])
        model = Model(inputs = [input_img, input_mask], outputs = [concat_output_img])
        print("====Generator Summary===")
        model.summary()
        return model 

    # =================================================================================== #
    #               7. Define the discriminator                                           #
    # =================================================================================== # 
   
    def build_discriminator(self):
        input_img = Input(shape=(self.missing_shape), dtype='float32', name='d_input')
        
        dis = (Conv2D(self.df, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))(input_img)
        dis = (LeakyReLU(alpha=0.2))(dis)
        dis = (Dropout(0.25))(dis)
        dis = (Conv2D(self.df*2, kernel_size=3, strides=2,  padding="same"))(dis)
        dis = (ZeroPadding2D(padding=((0,1),(0,1))))(dis)
        dis = (BatchNormalization(momentum=0.8))(dis)
        dis = (LeakyReLU(alpha=0.2))(dis)
        dis = (Dropout(0.25))(dis)
        dis = (Conv2D(self.df*4, kernel_size=3, strides=2,  padding="same"))(dis)
        dis = (BatchNormalization(momentum=0.8))(dis)
        dis = (LeakyReLU(alpha=0.2))(dis)
        dis = (Dropout(0.25))(dis)
        dis = (Conv2D(self.df*8, kernel_size=3, strides=2,  padding="same"))(dis)
        dis = (BatchNormalization(momentum=0.8))(dis)
        dis = (LeakyReLU(alpha=0.2))(dis)
        dis = (Dropout(0.25))(dis)
        dis = (Flatten())(dis)
        dis = (Dense(1))(dis)   
        
        model = Model(inputs=[input_img], outputs=dis)
        print("====Discriminator Summary===")
        model.summary()
        return model       

    # =================================================================================== #
    #               8. Define the train function                                          #
    # =================================================================================== #  
        
    def train(self):
            
        # Ground truths for adversarial loss
        valid = np.ones([self.batch_size, 1])
        fake = np.zeros((self.batch_size, 1))
        
        #Create arrays to store losses
        avg_d_loss=[]
        avg_g_loss=[]
        
        #Load data  
        preprocess_images = lambda x: (x - 127.5) / 127.5
        img_generator = ImageDataGenerator(rotation_range=15,horizontal_flip=True,preprocessing_function=preprocess_images)
        img_gen = img_generator.flow_from_directory(self.train_img_dir, 
                                                    target_size=(self.img_width, self.img_height), 
                                                    class_mode=None,
                                                    batch_size=self.batch_size, 
                                                    shuffle=True,  
                                                    interpolation="nearest", 
                                                    seed=123)
        
        preprocess_mask = lambda x: x * 1. / 255
        mask_generator = ImageDataGenerator(preprocessing_function=preprocess_mask)
        mask_gen = mask_generator.flow_from_directory(self.train_mask_dir, 
                                                      target_size=(self.img_width, self.img_height), 
                                                      class_mode=None,
                                                      batch_size=self.batch_size, 
                                                      color_mode="grayscale", 
                                                      shuffle=True,  
                                                      interpolation="nearest", 
                                                      seed=123)
        
        gc.collect() 
        num_samples = img_gen.samples
        wgan_num_steps = int(num_samples//self.batch_size)
        print("Number of WGAN steps {}".format(wgan_num_steps))
        global_step = 0
        
        discriminator_loss = []
        generator_loss = []
        
        init = tf.global_variables_initializer()
        with tf.device("gpu:0"):
            print("tf.keras code in this scope will run on GPU")        
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
            sess.run(init)
        if self.continue_train:
            #Edit last_trained_epoch in config.ini
            self.generator.load_weights(r'../{}/{}/weight_{}.h5'.format(self.models_path, self.dataset_name, self.last_trained_epoch))
            print ( "Successfully loaded last check point" )
        else:
            print ( "Failed to get check point" )                
        
        for epoch in range(self.num_epochs):
            
          batch_start_time = datetime.datetime.now() 
          step = 0
          for real_img in img_gen:
              mask = next(mask_gen)
              if step == wgan_num_steps:
                  break
              step +=1

    # =================================================================================== #
    #                             8.1. Predict images during training                     #
    # =================================================================================== #   
    
              gen_imgs=self.generator.predict([real_img,mask], self.batch_size) 
              gen_imgs = gen_imgs[:,:,:,0:3]        
              
    # =================================================================================== #
    #                             8.2. Train the discriminator                            #
    # =================================================================================== # 
    
              d_loss_real = self.discriminator.train_on_batch(real_img, valid)
              d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
              d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
              discriminator_loss.append(d_loss)
              avg_d_loss.append(np.mean(discriminator_loss))
    
    # =================================================================================== #
    #                             8.3. Train the generator                                #
    # =================================================================================== #
     
              g_loss = self.combined.train_on_batch([real_img ,mask], [real_img, valid])
              generator_loss.append(g_loss)
              avg_g_loss.append(np.mean(generator_loss))
              elapsed_time = datetime.datetime.now() - batch_start_time    
                 
    # =================================================================================== #
    #                             8.4. Plot the progress                                  #
    # =================================================================================== #
    
              gc.collect()   
              if global_step % self.sample_interval == 0:  
                if not os.path.exists("{}/{}/".format(self.models_path, self.dataset_name)):
                    os.makedirs("{}/{}/".format(self.models_path, self.dataset_name))
                print("[Epoch %d/%d] [Global Steps: %d/%d] [WGAN Steps: %d ] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f] time: %s " \
                          % (epoch, self.num_epochs,step,global_step,wgan_num_steps,
                             d_loss[0], 100 * d_loss[1],
                             g_loss[0],
                             np.mean(g_loss[1:3]),
                             elapsed_time))     
                print("Seen so far: %s samples" % ((global_step + 1) * self.batch_size)) 
                name = "{}/{}/weight_{}.h5".format(self.models_path, self.dataset_name, epoch+self.current_epoch)
                self.generator.save_weights(name)
              global_step += 1

    # =================================================================================== #
    #               9. Sample images during training                                      #
    # =================================================================================== # 
        
    def sample_images(self, dataset_name,input_img, sample_pred, mask, epoch):
        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)
        input_img = np.expand_dims(input_img[0], 0)
        input_mask = np.expand_dims(mask[0], 0)
        maskedImg = ((1 - input_mask)*input_img) + input_mask       
        img = np.concatenate(((maskedImg[0][:, :, ::-1]* 127.5 + 127.5).astype(np.uint8),
                              (sample_pred[0][:, :, ::-1]* 127.5 + 127.5).astype(np.uint8),
                              (input_img[0][:, :, ::-1]* 127.5 + 127.5).astype(np.uint8)),axis=1)
        img_filepath = os.path.join(self.dataset_name, 'pred_{}.jpg'.format(epoch+self.current_epoch))

        cv2.imwrite(img_filepath, img) 

    # =================================================================================== #
    #               10. Plot the discriminator and generator losses                       #
    # =================================================================================== # 
        
    def plot_logs(self,epoch, avg_d_loss, avg_g_loss):
        if not os.path.exists("LogsUnet"):
            os.makedirs("LogsUnet")
        plt.figure()
        plt.plot(range(len(avg_d_loss)), avg_d_loss,
                 color='red', label='Discriminator loss')
        plt.plot(range(len(avg_g_loss)), avg_g_loss,
                 color='blue', label='Adversarial loss')
        plt.title('Discriminator and Adversarial loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (Adversarial/Discriminator)')
        plt.legend()
        plt.savefig("LogsUnet/{}_paper/log_ep{}.pdf".format(self.dataset_name, epoch+self.current_epoch))


          

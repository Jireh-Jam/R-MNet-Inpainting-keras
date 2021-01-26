# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:24:02 2021

@author: Jireh Jam
"""

from configparser import ConfigParser
class MainConfig:
  
  def __init__(self, file_path):
    inpaint_config = ConfigParser()
    seen = inpaint_config.read(file_path)
    if not seen:
        raise ValueError('No config file found!')    
    print(inpaint_config.sections())
    self.testing = TestingConfig(inpaint_config['TESTING'])
    self.training = TrainingConfig(inpaint_config['TRAINING'])


class TestingConfig:
  
  def __init__(self, test_section):
    self.batch_size = int(test_section['BATCH_SIZE'])
    self.img_height = int(test_section['IMG_HEIGHT'])
    self.img_width = int(test_section['IMG_WIDTH'])
    self.channels = int(test_section['CHANNELS'])
    self.mask_channels = int(test_section['MASK_CHANNELS'])
    self.imgs_in_batch = int(test_section['IMGS_IN_BATCH'])
    self.last_img_on =  int(test_section['LAST_IMG_ON'])
    self.num_epochs = int(test_section['NUM_EPOCHS'])
    self.g_learning_rate = float(test_section['GEN_LEARNING_RATE'])
    self.d_learning_rate = float(test_section['DISC_LEARNING_RATE'])
    self.gf = int(test_section['GEN_FILTER'])
    self.df = int(test_section['DISC_FILTER'])
    self.current_epoch = int(test_section['CURRNT_EPOCH'])
    self.sample_interval = int(test_section['SAMPLE_INTERVAL'])
    self.beta_1 = float(test_section['BETA_1'])
    self.beta_2 = float(test_section['BETA_2'])
    self.epsilon = float(test_section['EPSILON'])
    self.last_trained_epoch = int(test_section['LAST_TRAINED_EPOCH'])
    
class TrainingConfig:
  
  def __init__(self, training_section):
    self.batch_size = int(training_section['BATCH_SIZE'])
    self.img_height = int(training_section['IMG_HEIGHT'])
    self.img_width = int(training_section['IMG_WIDTH']) 
    self.channels = int(training_section['NUM_CHANNELS'])  
    self.mask_channels = int(training_section['MASK_CHANNELS'])    
    self.num_epochs = int(training_section['NUM_EPOCHS'])
    self.g_learning_rate = float(training_section['GEN_LEARNING_RATE'])
    self.d_learning_rate = float(training_section['DISC_LEARNING_RATE'])
    self.gf = int(training_section['GEN_FILTER'])
    self.df = int(training_section['DISC_FILTER'])
    self.current_epoch = int(training_section['CURRNT_EPOCH'])
    self.sample_interval = int(training_section['SAMPLE_INTERVAL'])
    self.beta_1 = float(training_section['BETA_1'])
    self.beta_2 = float(training_section['BETA_2'])
    self.epsilon = float(training_section['EPSILON'])
    self.last_trained_epoch = int(training_section['LAST_TRAINED_EPOCH'])




 
    
    

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:57:01 2021

@author: Jireh Jam
"""

from rmnet import RMNETWGAN
from config import config

CONFIG_FILE = './config/config.ini'
config = config.MainConfig(CONFIG_FILE).training

if __name__ == '__main__':
    r_mnetwgan = RMNETWGAN(config)
    r_mnetwgan.train()

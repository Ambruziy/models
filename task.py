from WGAN import WGAN
import argparse

import tensorflow as tf
import os
import time
import re
from glob import glob
import numpy as np
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--num_epochs',
        help='number of epochs',
        type=int, default=100
    )

    parser.add_argument(
        '--batch_size',
        help='size of one batch',
        type=int, default=32
    )

    parser.add_argument(
        '--save_step',
        help='steps by save model',
        type=int, default=20
    )

    parser.add_argument(
        '--path_save',
        help='path, where model will be saved',
        type=str,
        required=True
    )

    parser.add_argument(
        '--path_data',
        help='path with data',
        type=str,
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__


    with tf.Session() as sess:
        wgan = WGAN(sess, num_epochs=arguments['num_epochs'], save_step=arguments['save_step'], path_save=arguments['path_save'],
                    path_data=arguments['path_data'], batch_size=arguments['batch_size'])
        wgan.train()

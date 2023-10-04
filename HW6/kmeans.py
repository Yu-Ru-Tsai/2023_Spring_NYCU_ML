import os
import numpy as np
import argparse
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='image1.png')
parser.add_argument('--input_name', type=str, default='image1')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--s', type=float, default=0.00001)
parser.add_argument('--c', type=float, default=0.0001)
parser.add_argument('--mode', type=str, default='kmeans++')
args = parser.parse_args()

if __name__=='__main__':
    output_dir = os.path.join('Kmeans', args.mode, 'k_{}'.format(args.k), args.input_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    X_color = load_png(args.input) 
    kernel = RBF_kernel(X_color, args.s, args.c) 
    cluster = kmeans(args.k, kernel, output_dir, args.mode)

    # GIF 
    GIF_dir = os.path.join(f'Kmeans_GIF', args.mode, 'k_{}'.format(args.k)) 
    if not os.path.exists(GIF_dir):
        os.mkdir(GIF_dir)
    create_gif(output_dir, GIF_dir, args.input_name)


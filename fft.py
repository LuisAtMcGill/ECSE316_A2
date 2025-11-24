import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='FFT Assignment')
parser.add_argument('-m', '--mode', type=int, default=1,
                    choices=[1, 2, 3, 4],
                    help='Operation mode: [1] Fast Mode, [2] Denoise, [3] Compress, [4] Plot Runtime')
parser.add_argument('-i', '--image', type=str, default='moonlanding.png',
                    help='Path to the input image file to process')
args = parser.parse_args()

mode, image_path = args.mode, args.image

# TODO : Implement the following functions
def naive_dft(signal):
    pass

def fft(signal):
    pass

def ifft(signal):
    pass

def fft_2d(image):
    pass

def ifft_2d(image):
    pass

# Mode 1: Fast Mode
def run_fast_mode(image_path):
    pass

# Mode 2: Denoise
def run_denoise(image_path):
    pass

# Mode 3: Compress
def run_compress(image_path):
    pass

# Mode 4: Plot Runtime
def run_plot_runtime():
    pass

# TODO : remove this print statement
print(f"Running mode {mode} on image {image_path}")

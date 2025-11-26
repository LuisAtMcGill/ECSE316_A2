import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse
from PIL import Image
import math


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
    N = signal.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, signal)

def naive_ift(signal):
    N = signal.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, signal) / N

def fft(signal):
    threshold = 16
    N = signal.shape[0]
    if N % 2 > 0 or N < threshold:
        return naive_dft(signal)  # Use normal DFT once split up enough
    else:
        even = naive_dft(signal[::2])
        odd = naive_dft(signal[1::2])
        form = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + form[:int(N / 2)] * odd,
                               even + form[int(N / 2):] * odd])
    

def ifft(signal):
    def helper(signal): # Helper to handle the algorithm
        threshold = 16
        N = signal.shape[0]
        if N <= threshold:
            return naive_ift(signal) / N
        else:
            even = helper(signal[::2])
            odd = helper(signal[1::2])
            form = np.exp(2j * np.pi * np.arange(N) / N)
            return np.concatenate([even + form[:int(N / 2)] * odd,
                                   even + form[int(N / 2):] * odd])

    y = helper(signal)
    N = signal.shape[0]
    return y / N

def fft_2d(image):
    M, N = image.shape
    new_img = np.zeros((M, N), dtype=np.complex64)

    for k in range(M):
        new_img[k] = fft(image[k])
    for l in range(N):
        new_img[:, l] = fft(new_img[:, l])
    return new_img

def ifft_2d(image):
    M, N = image.shape
    new_img = np.zeros((M, N), dtype=np.complex64)
    for k in range(M):
        new_img[k] = ifft(image[k])
    for l in range(N):
        new_img[:, l] = ifft(new_img[:, l])
    return new_img

# Mode 1: Fast Mode
def run_fast_mode(image_path):
    img = pad_image(image_path)
    fft_img = fft_2d(img)

    img = plt.imread(image_path).astype(float)
    plt.subplot(1, 3, 1)
    plt.title("Original img")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("fft of img")
    plt.imshow(np.abs(fft_img), norm=colors.LogNorm())
    plt.show()

# Mode 2: Denoise
def run_denoise(image_path):
    img = Image.open(image_path)
    img = np.asarray(img).astype(float)
    padded_img = pad_image(image_path)
    fft_img = fft_2d(padded_img)

    # Filter high frequencies
    freq_filter = 0.15
    filtered_fft_img = fft_img.copy()
    width, height = fft_img.shape

    filtered_fft_img[int(freq_filter * width) : int((1 - freq_filter) * width), :] = 0
    filtered_fft_img[:, int(freq_filter * height) : int((1 - freq_filter) * height)] = 0

    inverse_fft_img = ifft_2d(filtered_fft_img).astype(float)

    img = plt.imread(image_path).astype(float)
    plt.subplot(1, 3, 1)
    plt.title("Original img")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Filtered img")
    plt.imshow(inverse_fft_img[:img.shape[0], :img.shape[1]])
    plt.show()

# Mode 3: Compress
def run_compress(image_path):
    img = Image.open(image_path)
    img = np.asarray(img).astype(float)
    height, width = img.shape
    padded_img = pad_image(image_path)
    fft_img = fft_2d(padded_img)

    img = plt.imread(image_path).astype(float)
    plt.subplot(2, 3, 1)
    plt.title("Original img")
    plt.imshow(img)

    plt.subplot(2, 3, 2)
    img1 = compress(fft_img, 75, height, width)
    plt.imshow(img1)

    plt.show()

def compress(fft_img, ratio, height, width): # Helper used multiple times by run_compress. Compresses img and returns it
    comp_img = fft_img.copy()
    threshold = np.percentile(abs(fft_img), ratio)
    comp_img[abs(fft_img) < threshold] = 0

    inverse_fft_img = ifft_2d(comp_img).astype(float)
    inverse_fft_img = inverse_fft_img[:height, :width]
    return inverse_fft_img

# Mode 4: Plot Runtime
def run_plot_runtime():
    pass

# Helper function for 
def pad_image(image):
    img = Image.open(image)
    img_array = np.asarray(img)

    # Get dimensions padded to nearest power of 2
    width = int(pow(2, math.ceil(math.log2(img_array.shape[0]))))
    height = int(pow(2, math.ceil(math.log2(img_array.shape[1]))))

    new_img = np.zeros((width, height))
    new_img[:img_array.shape[0], :img_array.shape[1]] = img_array

    return new_img

# TODO : remove this print statement
print(f"Running mode {mode} on image {image_path}")
if mode == 1:
    run_fast_mode(image_path)
elif mode == 2:
    run_denoise(image_path)
elif mode == 3:
    run_compress(image_path)
print(f"Running mode {mode} on image {image_path}")

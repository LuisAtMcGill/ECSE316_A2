import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse
from PIL import Image
import math
import time


def naive_dft(signal):
    N = signal.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, signal)

def naive_ift(signal):
    # Return the un-normalized inverse DFT (normalization by N is done once at top-level in ifft)
    N = signal.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, signal)

def fft(signal):
    # Recursive Cooley-Tukey FFT with small base case using naive DFT.
    threshold = 16
    N = signal.shape[0]
    # naive DFT for odd lengths or small sizes (consistent with ifft's threshold)
    if N % 2 > 0 or N <= threshold:
        return naive_dft(signal)  # Use normal DFT once split up enough
    else:
        # recursive calls on even/odd indexed samples
        even = fft(signal[::2])
        odd = fft(signal[1::2])
        # twiddle factors W_N^k for k = 0..N/2-1
        W = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        first_half = even + W * odd
        second_half = even - W * odd
        return np.concatenate([first_half, second_half])


def ifft(signal):
    def helper(signal): # Helper to handle the algorithm
        threshold = 16
        N = signal.shape[0]
        if N <= threshold:
            # return simple dft, the normalization will be done once at the end of ifft
            return naive_ift(signal)
        else:
            even = helper(signal[::2])
            odd = helper(signal[1::2])
            form = np.exp(2j * np.pi * np.arange(N // 2) / N)
            first_half = even + form * odd
            second_half = even - form * odd
            return np.concatenate([first_half, second_half])

    y = helper(signal)
    N = signal.shape[0]
    # divide once by the full length to get the properly normalized inverse
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
    copy_img = pad_image(image_path) # For comparing to built-in fft
    fft_img = fft_2d(img)

    img = plt.imread(image_path).astype(float)
    plt.subplot(1, 3, 1)
    plt.title("Original img")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("fft of img")
    plt.imshow(np.abs(fft_img), norm=colors.LogNorm(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("fft of img (built-in)")
    plt.imshow(np.abs(np.fft.fft2(copy_img)), norm=colors.LogNorm(), cmap='gray')

    plt.show()

# Mode 2: Denoise
def run_denoise(image_path):
    img = Image.open(image_path).convert('L')
    img = np.asarray(img).astype(float)
    padded_img = pad_image(image_path)
    fft_img = fft_2d(padded_img)

    # Filter high frequencies
    freq_filter = 0.2
    filtered_fft_img = fft_img.copy()
    width, height = fft_img.shape

    filtered_fft_img[int(freq_filter * width) : int((1 - freq_filter) * width), :] = 0
    filtered_fft_img[:, int(freq_filter * height) : int((1 - freq_filter) * height)] = 0

    # Print number of non-zeros and fraction as per assignment instructions
    total_coeffs = fft_img.size
    nonzeros = np.count_nonzero(filtered_fft_img)
    print('Denoise: non-zero coefficients = {}/{} (fraction {:.4f})'.format(nonzeros, total_coeffs, nonzeros / float(total_coeffs)))

    inverse_fft_img = ifft_2d(filtered_fft_img).astype(float)

    img = plt.imread(image_path).astype(float)
    plt.subplot(1, 2, 1)
    plt.title("Original img")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title(f"Filtered img : {freq_filter}")
    plt.imshow(inverse_fft_img[:img.shape[0], :img.shape[1]], cmap='gray')
    plt.show()

# Mode 3: Compress
def run_compress(image_path):
    img = Image.open(image_path).convert('L')
    img = np.asarray(img).astype(float)
    height, width = img.shape
    padded_img = pad_image(image_path)
    fft_img = fft_2d(padded_img)

    # Prepare six compression levels (percentiles)
    percentiles = [0, 50, 75, 90, 99, 99.9]
    comp_images = []

    for p in percentiles:
        inverse, nz = compress(fft_img, p, height, width)
        comp_images.append(inverse)
        print('Compress: percentile {} -> non-zeros = {} (fraction {:.6f})'.format(p, nz, nz / float(fft_img.size)))

    # Display 2x3 subplot: show the 6 compressed results (P=0..99.9)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(6):
        axes[i].set_title('P={}'.format(percentiles[i]))
        axes[i].imshow(comp_images[i], cmap='gray')
    plt.tight_layout()
    plt.show()

# Helper used multiple times by run_compress. It zeroes out frequencies above a threshold
def compress(fft_img, ratio, height, width):
    comp_img = fft_img.copy()
    if ratio <= 0:
        threshold = -1
    else:
        threshold = np.percentile(abs(fft_img), ratio)
    if threshold >= 0:
        comp_img[abs(fft_img) < threshold] = 0

    non_zeros = np.count_nonzero(comp_img)

    inverse_fft_img = ifft_2d(comp_img).astype(float)
    inverse_fft_img = inverse_fft_img[:height, :width]
    return inverse_fft_img, non_zeros

# Mode 4: Plot Runtime
def run_plot_runtime():
    # Benchmark naive_dft vs fft for increasing sizes. Print mean and variance and plot them.
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    fft_means = []
    fft_vars = []
    dft_means = []
    dft_vars = []

    # "re-run the experiment at least 10 times" as per assignment instructions
    trials = 10

    for N in sizes:
        fft_times = []
        dft_times = []

        print(f"Running size {N}...")  # progress indicator

        for _ in range(trials):
            # Generate random 1D signal
            x = np.random.random(N)

            # Time FFT
            t0 = time.perf_counter()
            fft(x)
            t1 = time.perf_counter()
            fft_times.append(t1 - t0)

            # Time Naive DFT (we should only run this for small N, or it can take forever)
            # Naive 1024 can be really slow...

            x = np.random.random(N)
            t0 = time.perf_counter()
            naive_dft(x)
            t1 = time.perf_counter()
            dft_times.append(t1 - t0)


        fft_means.append(np.mean(fft_times))
        fft_vars.append(np.var(fft_times))

        dft_means.append(np.mean(dft_times))
        dft_vars.append(np.var(dft_times))

        print(f'Size {N}: FFT mean {fft_means[-1]:.6f}s | DFT mean {dft_means[-1]:.6f}s')
        print(f'\t   FFT std. dev {math.sqrt(fft_vars[-1]):.6f} | DFT std. dev {math.sqrt(dft_vars[-1]):.6f}')

    # Plot means with error bars
    # "twice the standard deviation" for 95-97% confidence
    fft_std = 2 * np.sqrt(fft_vars)
    dft_std = 2 * np.sqrt(dft_vars)

    plt.figure()
    plt.errorbar(sizes, fft_means, yerr=fft_std, label='FFT', marker='o', capsize=5)
    plt.errorbar(sizes, dft_means, yerr=dft_std, label='Naive DFT', marker='o', capsize=5)

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Signal size N (log scale)')
    plt.ylabel('Time (s, log scale)')
    plt.title('Runtime comparison: FFT vs Naive DFT (95% CI)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

# Helper function that pads image to prevent error
def pad_image(image):
    img = Image.open(image).convert('L')
    img_array = np.asarray(img)

    # Get dimensions padded to nearest power of 2
    width = int(pow(2, math.ceil(math.log2(img_array.shape[0]))))
    height = int(pow(2, math.ceil(math.log2(img_array.shape[1]))))

    new_img = np.zeros((width, height))
    new_img[:img_array.shape[0], :img_array.shape[1]] = img_array

    return new_img


# Move script execution into a main guard so importing is side-effect free
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFT Assignment')
    parser.add_argument('-m', '--mode', type=int, default=1,
                        choices=[1, 2, 3, 4],
                        help='Operation mode: [1] Fast Mode, [2] Denoise, [3] Compress, [4] Plot Runtime')
    parser.add_argument('-i', '--image', type=str, default='moonlanding.png',
                        help='Path to the input image file to process')
    args = parser.parse_args()

    mode, image_path = args.mode, args.image

    print('Running mode {} on image {}'.format(mode, image_path))
    if mode == 1:
        run_fast_mode(image_path)
    elif mode == 2:
        run_denoise(image_path)
    elif mode == 3:
        run_compress(image_path)
    elif mode == 4:
        run_plot_runtime()


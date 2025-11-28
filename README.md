# ECSE316_A2 - Fast Fourier Transform

## Group Members:
- Filip Piekarek - 260805461
- Luis Jarquin Romero - 261098623

## Required Source Files:
- fft.py
- moonlanding.jpg (default image)

## Python Version:
This code is compatible with Python 3.x.

## Required Libraries:
- numpy
- matplotlib
- PIL (Pillow)

## Example Usage:
```
python fft.py [-m mode] [-i image]
```

## Arguments:
- `-m mode`: Specifies the mode of operation. Options are:
  * `[1]` (default) : Convert image to its FFT form and display (Original vs. Log-scaled FFT).
  * `[2]`: Denoise the image using FFT/IFFT and display (Original vs. Denoised).
  * `[3]`: Display a 2x3 subplot of the image at 6 compression levels.
  * `[4]`: Compute and plot runtime complexity of FFT vs DFT for increasing image sizes.
- `-i image` (optional) : Path to the input image file. Default is `moonlanding.jpg`.

### Note: it could take a while to run modes 3 and 4 depending on the image size.

### Note: if the dependencies are not installed, you can install them using pip:
```
pip install matplotlib
```
```
pip install numpy
```
```
pip install Pillow
```

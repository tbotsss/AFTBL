# This code generates a spectrogram from an audio file using the librosa and matplotlib libraries.
# Defines a color map, loads the audio file, performs the Short-Time Fourier Transform (STFT),
# converts the amplitude matrix to decibels, resamples the spectrogram, scales the values, assigns colors to each value,
# creates an image from the data array, flips the image, saves it as 'spectrogram.png', and displays it.

# Import the necessary libraries
import matplotlib.colors as mcolors # we import the mcolors module from the matplotlib.colors library
import librosa # we import the librosa library to analyse the audio
import numpy as np # we import the numpy library to work with arrays
from PIL import Image # we import the Image module from the PIL library to work with images
from PIL import ImageOps # we import the ImageOps module from the PIL library to work with images
from scipy.signal import resample # we import the resample function from the scipy.signal library to resample the spectrogram
from sklearn.preprocessing import MinMaxScaler # we import the MinMaxScaler class from the sklearn.preprocessing library to rescale the spectrogram

height_in_pixels = 14 # Set the height of the spectrogram in pixels

colors = ["#5fbb46", "#adca40", "#4671b7", "#a0a579", "#fad93a", "#8f6279", "#d7523b"] # Define your color map using HEX codes for each colour

cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors) # create a linear segmented color map from the list of colors, called my_cmap. we don't use the my_cmap string (name), because we save the cmap object in a variable called cmap

y, sr = librosa.load(r'D:\RCA Year 2\AFTBL\audio\Act 1 MS.wav')# Load the audio file using librosa, which gives us the audio time series (y) and the sample rate (sr)

s = np.abs(librosa.stft(y, center=False, hop_length=2048)) # Run the Short-Time Fourier Transform (STFT) on the audio time series, which gives us a complex-valued matrix (s) of amplitudes of frequencies in time series. Increase hop_length for a blockier graph

s = librosa.amplitude_to_db(s, ref=np.max) # Convert the amplitude matrix to decibels which makes it logarithmic

s_resampled = resample(s, height_in_pixels, axis=0) # Resample the spectrogram to a lower resolution using scipy.signal, you can adjust the number to change the resolution

scaler = MinMaxScaler(feature_range=(0, 1))# Rescale s_resampled values from dB to 0-1 using scikit-learn's MinMaxScaler
s_resampled_scaled = scaler.fit_transform(s_resampled)


data = np.zeros((s_resampled_scaled.shape[0], s_resampled_scaled.shape[1], 4))# Add another dimension to s_resampled_scaled to make it three dimensional first value we use as the height of the values, the second value we use as the width of the values, and the third value we use as the color of the values


# Loop through the values of s_resampled_scaled and assign a color to each value
for j in range(s_resampled_scaled.shape[1]): # Loop through the columns
    for i in range(s_resampled_scaled.shape[0]): # Loop through the rows
        rgba_color = cmap(s_resampled_scaled[i, j])  # Get the RGBA color from the cmap
        data[i, j] = rgba_color # Assign the color to the corresponding value in data


# Create an image from the data array
img = Image.fromarray((data * 255).astype(np.uint8), 'RGBA')
img = ImageOps.flip(img)
img.save(r'D:\RCA Year 2\MS\RAGE.png')
img.show()

import rawpy
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from scipy.ndimage import gaussian_filter
import pickle
import tifffile as tiff
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.restoration import inpaint
from skimage.morphology import dilation, erosion
import cv2
import scipy
import scipy.stats

from interpolate_flats import *

def main():
	folder = 'K:/Orion_600mm/calibrated_lights_tiffs_optimized'


	filenames = list(filter(lambda s: s.endswith('.tif'), os.listdir(folder)))[:20]

	for channel in range(4):
		channel_all_values = np.zeros((len(filenames), 2012, 3012), dtype=np.float32)

		for i, fn in enumerate(tqdm.tqdm(filenames)):

			image = load_gray_tiff(os.path.join(folder, fn))
			image_rgb = extract_channel_image(image)

			img_channel = image_rgb[channel]
			mean = np.mean(img_channel)
			std = np.std(img_channel)

			n = 2
			low = mean - n*std
			high = mean + n*std

			nan_channel = img_channel.copy()
			nan_channel[np.where(img_channel < low)] = np.nan
			nan_channel[np.where(img_channel > high)] = np.nan

			channel_all_values[i] = nan_channel

		reg_mean_img = np.nanmean(channel_all_values, axis=0)

		plt.imshow(reg_mean_img)
		plt.show()

if __name__ == "__main__":
	main()
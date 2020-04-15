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

from optimize_flats_weighting import *
from interpolate_flats import *

def main():
	# darks_fn = 'K:/other_astro/iso800_30s_darks.tif'
	# bias_fn = 'K:/other_astro/bias_iso100.tif'
	# bias_fn = 'K:/other_astro/bias_iso800.tif'


	darks_fn = 'K:/other_astro/darks_iso100_135mm_30s.tif'
	bias_fn = 'K:/other_astro/bias_135mm_iso100.tif'

	if 0:
		img_gray = load_gray_tiff(bias_fn)
	elif 1:
		img_gray = load_gray_tiff(darks_fn)
	else:
		img_gray = load_gray_tiff(darks_fn) - load_gray_tiff(bias_fn)

		# img_gray = np.clip(img_gray, 0, 1)


	img_rgb = extract_channel_image(img_gray)

	for i, channel in enumerate(img_rgb):

		channel = remove_gradient(channel, quadratic=False)
		channel = gaussian_filter(channel, mode='nearest', sigma=5)

		plt.subplot(2, 2, i+1)
		z = 10
		low = np.percentile(channel, z)
		high = np.percentile(channel, 100 - z)

		plt.imshow(np.clip(channel, low, high))
		# plt.grid(True)

	plt.show()



	for i, channel in enumerate(img_rgb):

		# channel = remove_gradient(channel, quadratic=True)

		z = 0.1
		low = np.percentile(channel, z)
		high = np.percentile(channel, 100 - z)
		channel = np.clip(channel, low, high)

		plt.subplot(2, 2, i+1)
		plt.hist(channel.flatten(), bins = 1000)

	plt.show()

if __name__ == "__main__":
	main()
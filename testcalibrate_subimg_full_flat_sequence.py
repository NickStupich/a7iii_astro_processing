import os
import tifffile as tiff
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import astropy
from scipy.ndimage import gaussian_filter
import scipy.stats
import scipy.signal
import cv2
import functools
import cachetools
import datetime

import histogram_gap
from tiff_conversion import load_dark
from interpolate_fixed_flats import display_image, remove_gradient
from load_flats import load_flats_from_subfolders
import full_flat_sequence
import subimg_full_flat_sequence

def test():

	flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'		
	
	# test_folder = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats/4'
	test_folder = 'F:/2020/2020-04-11/flats_135mm_tshirtwall'
	bias_folder = 'K:/orion_135mm_bothnights/bias'

	bias_img = load_dark(bias_folder)

	for img_fn in os.listdir(test_folder):
		img = histogram_gap.load_raw_image(os.path.join(test_folder, img_fn), master_dark = bias_img)
		img = np.transpose(img, (2, 0, 1))

		flat_img = subimg_full_flat_sequence.get_subimg_matched_flat2(None, img, flats_progression_folder, bias_img, downsize_factor=4)

		calibrated_test_img = img / flat_img
		# calibrated_test_img = img


		for channel in range(4):
			calibrated_channel = calibrated_test_img[channel]
		
			calibrated_channel = remove_gradient(calibrated_channel, quadratic=True)
			calibrated_channel /= np.mean(calibrated_channel)

			overall_shape = gaussian_filter(calibrated_channel, sigma=50)

			calibrated_channel -= overall_shape

			std_dev = np.std(calibrated_channel)
			print(std_dev)
			plt.subplot(2, 1, 1)
			plt.imshow(np.clip(calibrated_channel, -std_dev, std_dev))
			plt.title(str(std_dev))

			plt.subplot(2, 1, 2)
			plt.imshow(overall_shape)

			plt.show()

if __name__ == "__main__":
	test()
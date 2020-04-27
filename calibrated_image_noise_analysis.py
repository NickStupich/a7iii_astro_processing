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
from interpolate_flats import load_gray_tiff, extract_channel_image, flatten_channel_image
from load_flats import load_flats_from_subfolders
import full_flat_sequence
import subimg_full_flat_sequence


def test():
	folder = 'K:/orion_135mm_bothnights/lights_out'
	tiffs_fns = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.tif'), os.listdir(folder))))
	print(tiffs_fns)
	for tiff_fn in tiffs_fns:
		img = tiff.imread(tiff_fn)
		img_rgb = extract_channel_image(img)

		for channel_index in range(img_rgb.shape[0]):
			channel_img = img_rgb[channel_index]

			channel_img = remove_gradient(channel_img, quadratic=True)
			channel_img -= np.mean(channel_img)

			if 0:
				z = 0.1
				plt.imshow(np.clip(channel_img, -z, z))
				plt.show()

			freq_img_complex = np.fft.rfft2(channel_img)

			if 0:
				plt.imshow(np.log(np.abs(freq_img_complex)))
				plt.show()

			plt.subplot(2, 1, 1)
			plt.semilogy(np.abs(freq_img_complex[0, :]))
			plt.grid(True)

			plt.subplot(2, 1, 2)
			plt.semilogy(np.abs(freq_img_complex[:, 0]))
			plt.grid(True)

			plt.show()



if __name__ == "__main__":
	test()
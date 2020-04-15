import rawpy
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from scipy.ndimage import gaussian_filter
import pickle

from analyze_bias import load_images, show_bands

# images_folder = 'images/bias'
darks_folder = 'images/darks_iso800_30s'
# images_folder = 'images/lights_nolens'
# flats_folder = 'images/flats_600mm'
# flats_folder = 'images/flats_600mm_2'

flats_folder = 'F:/2020/2020-02-15/flats_4'

dark_flats_folder = 'F:/2020/2020-02-15/dark_flats_iso800_1000'

# images_folder = 'F:/2018/2018-02-01/andromeda_600mm'
images_folder = 'F:/2020/2020-02-08/pleiades_600mm'

def main():
	dark_flats = load_images(dark_flats_folder)
	flat_frame = load_images(flats_folder)
	img_frame = load_images(images_folder)
	dark_frame = load_images(darks_folder)

	corrected_image = (img_frame - 0*dark_frame) / (flat_frame - 0*dark_flats)
	# corrected_image = img_frame - flat_frame*0.5
	corrected_image = np.clip(corrected_image, 0, np.inf)

	# plt.subplot(1, 3, 1)
	# plt.hist(dark_flats[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
	# plt.hist(dark_flats[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
	# plt.hist(dark_flats[:, :, 2].flatten(), bins = 100, color='b', histtype='step')
	# plt.yscale('log', nonposy='clip')

	plt.subplot(1, 3, 1)
	plt.hist(flat_frame[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
	plt.hist(flat_frame[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
	plt.hist(flat_frame[:, :, 2].flatten(), bins = 100, color='b', histtype='step')
	plt.yscale('log', nonposy='clip')

	plt.subplot(1, 3, 2)
	plt.hist(img_frame[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
	plt.hist(img_frame[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
	plt.hist(img_frame[:, :, 2].flatten(), bins = 100, color='b', histtype='step')
	plt.yscale('log', nonposy='clip')

	plt.subplot(1, 3, 3)
	plt.hist(corrected_image[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
	plt.hist(corrected_image[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
	plt.hist(corrected_image[:, :, 2].flatten(), bins = 100, color='b', histtype='step')
	plt.yscale('log', nonposy='clip')

	plt.show()


	show_bands(corrected_image)


if __name__ == "__main__":
	main()
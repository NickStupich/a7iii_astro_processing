import os
import tifffile as tiff
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import histogram_gap



def load_dark(folder):

	dark_cache_filename = os.path.join(folder, 'master_dark.tif')

	if os.path.exists(dark_cache_filename):
		print('found dark in cache')
		avg_dark = tiff.imread(dark_cache_filename)
		return avg_dark

	else:
		print('calculating dark image...')

		fns =  list(filter(lambda s: s.endswith('.ARW'), os.listdir(folder)))
		all_darks = None
		for i, fn in enumerate(tqdm.tqdm(fns)):
			full_fn = os.path.join(folder, fn)

			img = histogram_gap.read_raw_correct_hist(full_fn)

			if all_darks is None:
				all_darks = np.zeros((len(fns),) + img.shape, dtype=img.dtype)

			all_darks[i] = img

		if 0:
			for channel in range(4):
				all_pixels = all_darks[:, :, :, channel].flatten()

				plt.hist(all_pixels, bins = np.arange(0, np.max(all_pixels)+1))
				plt.grid(True)
				plt.show()

		avg_dark = np.mean(all_darks, axis=0)

		tiff.imwrite(dark_cache_filename, avg_dark.astype('float32'))

	return avg_dark

def flatten_channel_image(img):
	result = np.zeros((2*img.shape[0], 2*img.shape[1]))
	result[::2, ::2] = img[:, :, 0]
	result[1::2, ::2] = img[:, :, 1]
	result[::2, 1::2] = img[:, :, 2]
	result[1::2, 1::2] = img[:, :, 3]

	return result

def convert_raw_to_lights_folder(folder, bias_frame):
	print('fixing histograms in folder: ', folder)
	fns =  list(filter(lambda s: s.endswith('.ARW'), os.listdir(folder)))
	for fn in tqdm.tqdm(fns):
		full_fn = os.path.join(folder, fn)

		img = histogram_gap.read_raw_correct_hist(full_fn)

		if bias_frame is not None:
			img -= bias_frame
		else:
			img -= 512
			print('no bias frame')

		img = np.clip(img, 0, np.inf)

		# print(img.shape)

		flat_img = flatten_channel_image(img)

		tiff.imwrite(full_fn.rstrip('.ARW') + '_histfix.tif', flat_img.astype('float32'))

if __name__ == "__main__":

	flats_folder = 'K:/orion_135mm_histfix/flats_with_bias/5'
	lights_folder = 'K:/orion_135mm_histfix/lights_with_dark'

	darks_folder = 'K:/orion_135mm_histfix/darks'
	bias_folder = 'K:/orion_135mm_histfix/bias'


	bias_img = load_dark(bias_folder)
	convert_raw_to_lights_folder(flats_folder, bias_frame = bias_img)

	exit(0)

	dark_img = load_dark(darks_folder)
	convert_raw_to_lights_folder(lights_folder, bias_frame = dark_img)
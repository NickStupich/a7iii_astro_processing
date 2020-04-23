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
import scipy.stats
import astropy
import astropy.stats
from sklearn.neighbors import RadiusNeighborsRegressor
from datetime import datetime
import cachetools

#don't subtract black level
#do darks for real black level
#gaps confirmed down to lowest one

#without black sub: 576, 705, 834, 963, ~1220, ~1484 ....6256

IMG_MAX = 2**14

def fix_channel(channel):
	first_gap_index = 576
	gap_offset = 129

	channel_out = channel.copy()

	current_gap_index = first_gap_index
	while(current_gap_index < np.max(channel)):
		channel_out[np.where(channel_out >= current_gap_index)] -= 1

		current_gap_index = current_gap_index + gap_offset - 1

		# break

	return channel_out


lut = None
def fix_channel2(channel):
	global lut

	if lut is None:
		lut = np.arange(0, 2**14)
		lut = fix_channel(lut)

	result = lut[channel]

	return result

def read_raw_correct_hist(fn, plot=False):
	raw = rawpy.imread(fn)

	out = np.zeros((raw.raw_image_visible.shape[0]//2, raw.raw_image_visible.shape[1]//2, 4))
	# print('out.shape: ', out.shape)
	for channel in range(4):
		# raw_channel = raw.raw_image[channel//2::2, channel%2::2]
		offsets = (channel//2, channel%2)
		# offsets = np.unravel_index(np.argmax(raw.raw_pattern==channel), raw.raw_pattern.shape)
		raw_channel = raw.raw_image_visible[offsets[0]::2, offsets[1]::2]
		fixed_channel = fix_channel2(raw_channel)

		out[:, :, channel] = fixed_channel

		if plot:
			plt.subplot(2, 1, 1)
			plt.hist(raw_channel.flatten(), bins = np.arange(0, np.max(raw_channel)))
			plt.title('raw channel')

			plt.subplot(2, 1, 2)
			plt.hist(fixed_channel.flatten(), bins = np.arange(0, np.max(raw_channel)))
			plt.title('corrected channel')

			plt.show()

	return out

cache = cachetools.LRUCache(maxsize=32)
@cachetools.cached(cache=cache, key=lambda *args, **kwargs: cachetools.keys.hashkey(args[0]))
def load_raw_image(filename, master_dark = None):
	img = read_raw_correct_hist(filename)

	if master_dark is not None:
		img -= master_dark
	else:
		img -= 512
		# print('no bias/dark frame')

	img = np.clip(img, 0, np.inf)
	img /= IMG_MAX
	return img

if __name__ == "__main__":
	fn = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats/5/DSC03596.ARW'
	# fn = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats/1/DSC03478.ARW'
	# fn = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats/3/DSC03544.ARW'
	# fn = 'F:/Pictures/Lightroom/2020/2020-03-01/135mm_computer_screen_flats/1/DSC06590.ARW'
	# fn = 'F:/Pictures/Lightroom/2020/2020-03-07/darks_135mm_30s_iso100/DSC06751.ARW'
	if 0:
		raw = rawpy.imread(fn)
		
		print(raw.raw_image.shape)
		red = raw.raw_image[1::2, ::2] - raw.black_level_per_channel[0]
		# red = raw.raw_image[1::2, 1::2]
		plt.hist(red.flatten(), bins = np.arange(0, np.max(red))); plt.show()
	elif 0:
		read_raw_correct_hist(fn, plot=True)
	else:

		raw = rawpy.imread(fn)

		out = np.zeros((raw.raw_image.shape[0]//2, raw.raw_image.shape[1]//2, 4))

		for channel in range(4):
			raw_channel = raw.raw_image[channel//2::2, channel%2::2]
			start = datetime.now()
			fixed_channel_ref = fix_channel(raw_channel)
			print('ref implementation: ', (datetime.now() - start))
			start = datetime.now()
			fixed_channel = fix_channel2(raw_channel)
			print('new implementation: ', (datetime.now() - start))

			# if not np.equal(fixed_channel_ref.flatten(), fixed_channel.flatten()):
			if not (fixed_channel_ref == fixed_channel).all():
				print('***NOT EQUAL***')
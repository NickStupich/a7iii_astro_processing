import os
import tifffile as tiff
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import astropy
from scipy.ndimage import gaussian_filter
import scipy.stats
import cv2
import functools
import cachetools

import histogram_gap
from tiff_conversion import load_dark, IMG_MAX
from interpolate_flats import load_gray_tiff, extract_channel_image, flatten_channel_image
from interpolate_fixed_flats import get_exposure_matched_flat, display_image, remove_gradient
from full_image_calibration import load_flats_from_subfolders, load_raw_image, calc_relative_image_noise_level


# @functools.lru_cache()
def calc_flat_image_means(flat_images_rgb, proportiontocut = 0.2):
	proportiontocut = 0.2

	flat_image_means = np.mean(flat_images_rgb, axis=(2, 3))
	print('flat image means: ', flat_image_means)

	if 1:
		for channel in range(flat_images_rgb.shape[0]):
			for img in range(flat_images_rgb.shape[1]):
				flat_image_means[channel, img] = scipy.stats.trim_mean(flat_images_rgb[channel, img].flatten(), proportiontocut = proportiontocut)

		print('flat image means: ', flat_image_means)

	return flat_image_means

flat_image_means = None
def get_flat_and_bandingfix_flat(flat_images_rgb, test_img_rgb, flats_progression_folder, bias_img = None):

	proportiontocut = 0.2
	global flat_image_means
	if flat_image_means is None:
		flat_image_means = calc_flat_image_means(flat_images_rgb, proportiontocut)

	output_flat = np.zeros_like(test_img_rgb)
	for channel in range(test_img_rgb.shape[0]):
		# output_flat[channel] = 1; continue
		print('starting channel ', channel)
		test_img_channel_mean = scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = proportiontocut)
		print('test img mean: ', test_img_channel_mean)
		print(flat_image_means[:, channel] - test_img_channel_mean)
		closest_flat_index = np.abs(flat_image_means[:, channel] - test_img_channel_mean).argmin()

		print('closest flat index: ', closest_flat_index)

		daily_flat_channel = flat_images_rgb[closest_flat_index, channel, :, :]
		print('daily flat mean: ', np.mean(daily_flat_channel))

		banding_fix_flat = get_relative_flat(flats_progression_folder, test_img_rgb[channel], daily_flat_channel, channel, bias_img = bias_img)

		output_channel_flat = daily_flat_channel * banding_fix_flat

		output_channel_flat /= np.mean(output_channel_flat)

		print('output_channel_flat: ', np.mean(output_channel_flat))


		output_flat[channel] = output_channel_flat

	return output_flat

@cachetools.cached(cache={}, key=lambda folder, bias_img: cachetools.keys.hashkey(folder))
def get_progression_flat_mean(folder, bias_img = None):
	cache_name = os.path.join(folder, 'mean_frame.tiff')
	means_cache_name = os.path.join(folder, 'mean_brightnesses.npy')

	image_names = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(folder))))
	# print(image_names)

	if os.path.exists(cache_name):# and False:
		sum_image = tiff.imread(cache_name)
		all_image_means = np.load(means_cache_name)
	else:

		sum_image = None
		all_image_means = []

		for img_fn in tqdm.tqdm(image_names):
			# img = histogram_gap.read_raw_correct_hist(img_fn)
			img = load_raw_image(img_fn, bias_img) / IMG_MAX
			channel_means = np.mean(img, axis=(0,1))

			img = img / channel_means[np.newaxis, np.newaxis, :]

			all_image_means.append(channel_means)
			# print(channel_means)

			if sum_image is None:
				sum_image = img
			else:
				sum_image += img


		sum_image /= len(image_names)
		all_image_means = np.array(all_image_means)

		tiff.imwrite(cache_name, sum_image)
		np.save(means_cache_name, all_image_means)

	return sum_image, all_image_means, image_names

#Custom Decorator function
def listToTuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result
    return wrapper

# @listToTuple
# @functools.lru_cache()
@cachetools.cached(cache={}, key=lambda image_filenames, closest_index, half_images_to_average, channel, bias_img: cachetools.keys.hashkey(tuple(image_filenames), closest_index, half_images_to_average))
def load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img):
	# global bias_img

	if 0:
		# print(all_flat_channel_means[closest_index], input_channel_mean)
		flat_image = load_raw_image(image_filenames[closest_index], bias_img) / IMG_MAX
		flat_image_channel = flat_image[:, :, channel]
	else:
		flat_image_channel_stack = None
		for i, image_index in enumerate(range(closest_index - half_images_to_average, closest_index + half_images_to_average+1)):
			flat_image = load_raw_image(image_filenames[image_index], bias_img) /IMG_MAX
			flat_image_channel = flat_image[:, :, channel]

			if flat_image_channel_stack is None:
				flat_image_channel_stack = np.zeros((2*half_images_to_average+1,) + flat_image_channel.shape, dtype=flat_image_channel.dtype)

			flat_image_channel_stack[i] = flat_image_channel


		#todo: normalize image brightnesses before combining?
		flat_image_channel = np.mean(flat_image_channel_stack, axis=0)
			# print(flat_image.shape)

	return flat_image_channel

def get_flat_matching_brightness(folder, input_channel, channel, half_images_to_average = 3, bias_img = None):
	overall_mean, all_image_means, image_filenames = get_progression_flat_mean(folder, bias_img = bias_img)

	#todo: mean -> something more outlier resistant
	input_channel_mean = scipy.stats.trim_mean(input_channel.flatten(), proportiontocut = 0.2)

	all_flat_channel_means = all_image_means[:, channel]
	closest_index = np.abs(all_flat_channel_means - input_channel_mean).argmin()
	print('input channel  mean: ', input_channel_mean)
	print('closest index: ', closest_index, len(all_flat_channel_means))
	# plt.plot(all_flat_channel_means); plt.show()


	flat_image_channel = load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img)


	# for _ in range(3):
	ratios = input_channel_mean / flat_image_channel
	median_ratio = np.median(ratios)
	print('median ratio: ', median_ratio)

	input_channel_mean *= median_ratio

	closest_index = np.abs(all_flat_channel_means - input_channel_mean).argmin()
	print('closest index: ', closest_index)
	print(all_flat_channel_means[closest_index], input_channel_mean)
	flat_image = load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img)

	# noise_level = calc_relative_image_noise_level(flat_image_channel)
	# print('relative noise level of flat: ', noise_level)

	return flat_image_channel

def make_animation():
	folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'

	overall_mean, all_image_means, image_filenames = get_progression_flat_mean(folder)

	if 1:
		for channel in range(4):
			plt.plot(all_image_means[:, channel])

		plt.grid(True)
		plt.show()

		for channel in range(4):
			plt.imshow(overall_mean[:, :, channel])
			plt.title(str(channel))
			plt.show()

	if 0:
		size = (3024, 2012)
		channel_index = 0

		out = cv2.VideoWriter('channel_%d.avi' % channel_index,cv2.VideoWriter_fourcc(*'mp4v'), 15, size, isColor=False)

		image_names = list(filter(lambda s: s.endswith('.ARW'), os.listdir(folder)))#[::10]



		for img_fn in tqdm.tqdm(image_names):
			img = histogram_gap.read_raw_correct_hist(os.path.join(folder, img_fn))
			channel = img[:, :, channel_index]

			channel /= overall_mean[:, :, channel_index]

			channel_mean = np.mean(channel)
			channel /= channel_mean

			channel -=1
			scale = 10
			channel *= scale

			channel = np.clip(255*(channel + 0.5), 0, 255).astype(np.uint8)

			print(channel.shape)
			out.write(channel)	

		out.release()

def get_relative_flat(flats_progression_folder, test_channel, matched_flat_channel, channel_index, bias_img = None):
	half_images_to_average = 7
	matching_flat_1 = get_flat_matching_brightness(flats_progression_folder, test_channel, channel_index, bias_img = bias_img, half_images_to_average = half_images_to_average)
	matching_flat_2 = get_flat_matching_brightness(flats_progression_folder, matched_flat_channel, channel_index, bias_img = bias_img, half_images_to_average = half_images_to_average)

	#todo: center this around 0?
	#todo: spatial filtering at all?
	relative_flat = matching_flat_1 / matching_flat_2

	return relative_flat

def show_relative_flats(test_img_path, bias_img = None):
	test_img = load_raw_image(test_img_path, bias_img) / IMG_MAX
	print(test_img.shape)
	for channel in range(4):

		test_channel = test_img[:, :, channel]
		flat_channel = 1.1 * test_img[:, :, channel]

		relative_flat = get_relative_flat(flats_progression_folder, test_channel, flat_channel, channel, bias_img)

	# for channel in [3]:
		# matching_flat = get_flat_matching_brightness(flats_progression_folder, test_img[:, :, channel], channel, bias_img = bias_img)

		# test_img_offset = test_img * 1.1

		# matching_flat_offset = get_flat_matching_brightness(flats_progression_folder, test_img_offset[:, :, channel], channel, bias_img = bias_img)

		# relative_flat = matching_flat / matching_flat_offset

		# plt.imshow(relative_flat)
		display_image(relative_flat)
		plt.show()

if __name__ == "__main__":
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	master_bias = load_dark(bias_folder)
	

	# make_animation()
	flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'

	test_img_path = 'F:/Pictures/Lightroom/2020/2020-02-29/orion_600mm/DSC05669.ARW'
	show_relative_flats(test_img_path, master_bias)

	test_img = load_raw_image(test_img_path, None) / IMG_MAX

	for channel in range(test_img.shape[-1]):
		matching_flat = get_flat_matching_brightness(flats_progression_folder, test_img[:, :, channel], channel, half_images_to_average=3)

		plt.subplot(2, 2, 1)
		display_image(test_img[:, :, channel])
		plt.title('test img')

		plt.subplot(2, 2, 2)
		display_image(matching_flat)
		plt.title('matching flat')

		ratios = (test_img[:, :, channel] / matching_flat).flatten()

		plt.subplot(2, 2, 3)
		bin_range = 0.1
		bins = np.linspace(1 - bin_range, 1+bin_range, 1000)
		plt.hist(ratios, bins = bins)
		plt.yscale('log', nonposy='clip')
		plt.grid(True)

		plt.show()

	print(np.mean(test_img, axis=(0, 1)))
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
from tiff_conversion import load_dark
from interpolate_fixed_flats import display_image, remove_gradient
from load_flats import load_flats_from_subfolders

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

@cachetools.cached(cache={}, key=lambda *args, **kwargs: cachetools.keys.hashkey(args[0]))
def get_progression_flat_mean(folder, bias_img = None):
	cache_name = os.path.join(folder, 'mean_frame.tiff')
	means_cache_name = os.path.join(folder, 'mean_brightnesses.npy')

	image_names = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(folder))))

	#is this ok in general? probably?
	# def sortKey(s):
	# 	result = int(s.split(os.path.sep)[-1].strip('DSC').split('.')[-2])
	# 	if result > 9000: result -= 10000
	# 	return result

	# image_names.sort(key = sortKey)
	# print(image_names)

	if os.path.exists(cache_name):# and False:
		sum_image = tiff.imread(cache_name)
		all_image_means = np.load(means_cache_name)

	else:

		sum_image = None
		all_image_means = []

		for img_fn in tqdm.tqdm(image_names):
			img = histogram_gap.load_raw_image(img_fn, bias_img)
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
		# plt.plot(all_image_means)
		# plt.show()
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


cache = cachetools.LRUCache(maxsize=32)
@cachetools.cached(cache=cache, key=lambda image_filenames, closest_index, half_images_to_average, channel, bias_img: cachetools.keys.hashkey(tuple(image_filenames), closest_index, half_images_to_average))
def load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img):
	# global bias_img

	if 0:
		# print(all_flat_channel_means[closest_index], input_channel_mean)
		flat_image = histogram_gap.load_raw_image(image_filenames[closest_index], bias_img)
		flat_image_channel = flat_image[:, :, channel]
	else:
		if 0:
			flat_image_channel_stack = None
			for i, image_index in enumerate(range(max(0, closest_index - half_images_to_average), min(len(image_filenames)-1, closest_index + half_images_to_average+1))):
				flat_image = histogram_gap.load_raw_image(image_filenames[image_index], bias_img)
				flat_image_channel = flat_image[:, :, channel]

				if flat_image_channel_stack is None:
					flat_image_channel_stack = np.zeros((2*half_images_to_average+1,) + flat_image_channel.shape, dtype=flat_image_channel.dtype)

				flat_image_channel_stack[i] = flat_image_channel
		else:
			flat_image_channel_stack = []
			for i, image_index in enumerate(range(max(0, closest_index - half_images_to_average), min(len(image_filenames)-1, closest_index + half_images_to_average+1))):
				flat_image = histogram_gap.load_raw_image(image_filenames[image_index], bias_img)
				flat_image_channel = flat_image[:, :, channel]

				flat_image_channel_stack.append(flat_image_channel)
			flat_image_channel_stack = np.array(flat_image_channel_stack)

		#todo: normalize image brightnesses before combining?
		image_means = np.mean(flat_image_channel_stack, axis=(1,2))
		# print(image_means)
		for i in range(flat_image_channel_stack.shape[0]):
			# normalization_factor = image_means[half_images_to_average] / image_means[i]
			normalization_factor = image_means[image_means.shape[0]//2] / image_means[i]
			flat_image_channel_stack[i] *= normalization_factor
		

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
	print('closest progression flat index: ', closest_index, len(all_flat_channel_means))
	# plt.plot(all_flat_channel_means); plt.show()


	flat_image_channel = load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img)


	# for _ in range(3):
	ratios = input_channel_mean / flat_image_channel
	median_ratio = np.median(ratios)
	print('median ratio: ', median_ratio)

	input_channel_mean *= median_ratio

	closest_index = np.abs(all_flat_channel_means - input_channel_mean).argmin()
	print('closest progression flat index: ', closest_index, len(all_flat_channel_means))
	print(all_flat_channel_means[closest_index], input_channel_mean)
	flat_image_channel = load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img)

	# noise_level = calc_relative_image_noise_level(flat_image_channel)
	# print('relative noise level of flat: ', noise_level)

	return flat_image_channel


def get_flat_matching_brightness_histogram_match(folder, input_channel, channel, half_images_to_average = 3, bias_img = None):
	overall_mean, all_image_means, image_filenames = get_progression_flat_mean(folder, bias_img = bias_img)

	#todo: mean -> something more outlier resistant
	input_channel_mean = scipy.stats.trim_mean(input_channel.flatten(), proportiontocut = 0.2)

	all_flat_channel_means = all_image_means[:, channel]
	closest_index = np.abs(all_flat_channel_means - input_channel_mean).argmin()
	print('closest progression flat index: ', closest_index, len(all_flat_channel_means))

	flat_image_channel = load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img)


	for _ in range(3):

		for peak_range in [0.02, 0.05, 0.1, 0.2]:
			all_ratios = (input_channel / flat_image_channel).flatten()
			ratios = all_ratios[np.where(np.abs(all_ratios - 1) < peak_range)]
			n, bins = np.histogram(ratios, bins = np.linspace(1 - peak_range, 1 + peak_range, 1001))

			bins = (bins[1:] + bins[:-1])/2
			fit = np.polyfit(bins, n, deg=2)
			peak = -fit[1] / (2*fit[0])
			print(peak, fit)

			if np.isnan(peak) and False:
				plt.imshow(flat_image_channel)
				plt.show()

				plt.imshow(input_channel)
				plt.show()

			if np.abs(peak - 1) < peak_range:
				break

		print('peak value: ', peak)

		input_channel_mean *= peak

		closest_index = np.abs(all_flat_channel_means - input_channel_mean).argmin()
		print('closest index: ', closest_index)
		print(all_flat_channel_means[closest_index], input_channel_mean)
		flat_image_channel = load_flat_img(image_filenames, closest_index, half_images_to_average, channel, bias_img)

		# noise_level = calc_relative_image_noise_level(flat_image_channel)
		# print('relative noise level of flat: ', noise_level)

	return flat_image_channel

def make_animation():
	folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
	bias_img = None

	overall_mean, all_image_means, image_filenames = get_progression_flat_mean(folder, bias_img = bias_img)

	if 0:
		for channel in range(4):
			plt.plot(all_image_means[:, channel])

		plt.grid(True)
		plt.show()

		for channel in range(4):
			plt.imshow(overall_mean[:, :, channel])
			plt.title(str(channel))
			plt.show()

	if 1:
		size = (3012, 2012)
		channel_index = 0

		out = cv2.VideoWriter('channel_%d.avi' % channel_index,cv2.VideoWriter_fourcc(*'mp4v'), 15, size, isColor=False)

		image_names = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(folder))))#[::10]
		image_names = image_names[::-1]
		# print(image_names)


		for i, img_fn in enumerate(tqdm.tqdm(image_names)):
			if 0:
				img = histogram_gap.load_raw_image(os.path.join(folder, img_fn), master_dark = bias_img)
				channel = img[:, :, channel_index]
			else:
				channel = load_flat_img(image_names, i, half_images_to_average=7, channel=channel_index, bias_img=bias_img)


			channel /= overall_mean[:, :, channel_index]

			channel_mean = np.mean(channel)
			channel /= channel_mean

			channel -=1
			scale = 10
			channel *= scale

			channel = np.clip(255*(channel + 0.5), 0, 255).astype(np.uint8)

			out.write(channel)	

		out.release()

def get_relative_flat(flats_progression_folder, test_channel, matched_flat_channel, channel_index, bias_img = None):
	half_images_to_average = 7

	# matching_func = get_flat_matching_brightness
	matching_func = get_flat_matching_brightness_histogram_match

	matching_flat_1 = matching_func(flats_progression_folder, test_channel, channel_index, bias_img = bias_img, half_images_to_average = half_images_to_average)
	matching_flat_2 = matching_func(flats_progression_folder, matched_flat_channel, channel_index, bias_img = bias_img, half_images_to_average = half_images_to_average)

	#todo: spatial filtering at all?
	relative_flat = matching_flat_1 / matching_flat_2

	relative_flat = gaussian_filter(relative_flat, mode='nearest', sigma=7)
	print('blurred relative flat')

	return relative_flat

def show_relative_flats(test_img_path, bias_img = None):
	test_img = histogram_gap.load_raw_image(test_img_path, bias_img)
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

def test():
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	master_bias = load_dark(bias_folder)
	
	flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'

	test_img_path = 'F:/Pictures/Lightroom/2020/2020-02-29/orion_600mm/DSC05669.ARW'
	show_relative_flats(test_img_path, master_bias)

	test_img = histogram_gap.load_raw_image(test_img_path, None) 

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

def test2():
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	master_bias = load_dark(bias_folder)
	
	# flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
	flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'

	test_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/3'
	n_test_images = 3

	test_filenames = list(map(lambda s2: os.path.join(test_folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(test_folder))))

	for i in range(0, len(test_filenames), n_test_images):
		fns = test_filenames[i:i+n_test_images]

		test_imgs = [histogram_gap.load_raw_image(fn, master_bias) for fn in fns]

		mean_test_img = np.mean(test_imgs, axis=0)

		for channel in range(mean_test_img.shape[-1]):
			test_img_channel = mean_test_img[:, :, channel]

			# matching_flat = get_flat_matching_brightness(flats_progression_folder, test_img_channel, channel, half_images_to_average=7)
			matching_flat = get_flat_matching_brightness_histogram_match(flats_progression_folder, test_img_channel, channel, half_images_to_average=7)
			
			corrected_test_img = test_img_channel / matching_flat

			ratio_range = 0.05
			ratios = corrected_test_img[np.where(np.abs(corrected_test_img-1) < 0.05)].flatten()
			
			def display_image(img, z=1):
				disp_image = img.copy()
				disp_image = remove_gradient(disp_image)
				disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

				low = np.percentile(disp_image, z)
				high = np.percentile(disp_image, 100 - z)
				disp_image = np.clip(disp_image, low, high)
				plt.imshow(disp_image)

			plt.subplot(2, 2, 1)
			display_image(test_img_channel)
			plt.title('raw test img')

			plt.subplot(2, 2, 2)
			display_image(matching_flat)
			plt.title('matching flat')

			plt.subplot(2, 2, 3)
			display_image(corrected_test_img)
			plt.title('corrected test img')

			plt.subplot(2, 2, 4)
			plt.hist(ratios, bins = 1001)
			plt.grid(True)
			plt.title('flat : bright ratio img')

			plt.show()

if __name__ == "__main__":
	

	# make_animation()
	# test()

	test2()
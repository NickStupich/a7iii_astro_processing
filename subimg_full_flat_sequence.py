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

# cache_base = 'K:/cache/135mm'
cache_base = 'K:/cache/600mm'

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
		print(image_names)


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

def load_channel_progression(folder, bias_img, downsize_factor, channel_index):

	image_names = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(folder))))
	def sortKey(s):
		result = int(s.split(os.path.sep)[-1].strip('DSC').split('.')[-2])
		if result > 9000: result -= 10000
		return result

	image_names.sort(key = sortKey)
	print('# flat images: ', len(image_names))
	# print(image_names)

	cache_filename = cache_base + '/flat_channel_cache%d_%d.npy' % (channel_index, downsize_factor)

	if not os.path.exists(cache_filename):
		channel_cache = None

		for i, img_fn in enumerate(tqdm.tqdm(image_names)):
			if 1:
				img = histogram_gap.load_raw_image(os.path.join(folder, img_fn), master_dark = bias_img)
				channel = img[:, :, channel_index]
			else:
				channel = full_flat_sequence.load_flat_img(image_names, i, half_images_to_average=3, channel=channel_index, bias_img=bias_img)

			channel_downsized = channel.reshape((channel.shape[0]//downsize_factor, downsize_factor, channel.shape[1]//downsize_factor, downsize_factor)).mean(3).mean(1)

			if channel_cache is None:
				channel_cache = np.zeros((len(image_names), channel_downsized.shape[0], channel_downsized.shape[1]), dtype=np.float32)

			channel_cache[i] = channel_downsized

			if 0:
				plt.imshow(channel)
				plt.show()
				plt.imshow(channel_downsized)
				plt.show()


		np.save(cache_filename, channel_cache)
		print('saved flat channel cache')
	else:
		print('loading from cache file...')
		start = datetime.datetime.now()
		channel_cache = np.load(cache_filename)
		print('done', (datetime.datetime.now() - start))
		print(channel_cache.shape)

	return channel_cache

# cache = cachetools.LRUCache(maxsize=32)
# @cachetools.cached(cache=cache, key=lambda folder, bias_img, downsize_factor, channel_index, num_images_to_avg: cachetools.keys.hashkey(folder, downsize_factor, channel_index, num_images_to_avg))
def load_channel_progression_brightness_filtered(folder, bias_img, downsize_factor, channel_index, num_images_to_avg):

	cache_filename = cache_base + '/filtered_%d_flat_channel_cache%d_%d.npy' % (num_images_to_avg, channel_index, downsize_factor)

	if not os.path.exists(cache_filename):
		channel = load_channel_progression(folder, bias_img, downsize_factor, channel_index)
		print(channel.shape)

		filter_coefs = np.ones(num_images_to_avg) / num_images_to_avg

		print('filtering raw channel...')
		for yi in tqdm.tqdm(range(channel.shape[1])):
			for xi in range(channel.shape[2]):
				channel[:, yi, xi] = scipy.signal.filtfilt(filter_coefs, [1], channel[:, yi, xi])

		np.save(cache_filename, channel)

	else:
		print('loading from cache file...')
		start = datetime.datetime.now()
		channel = np.load(cache_filename)
		print('done', (datetime.datetime.now() - start))
		print(channel.shape)

	return channel, np.mean(channel, axis=(1,2))

def load_channel_progression_brightness_filtered_spatialfiltered(folder, bias_img, downsize_factor, channel_index, num_images_to_avg, spatial_sigma_raw = 10):

	cache_filename = cache_base + '/filtered_%d_flat_channel_cache%d_%d_spatial_filter%d.npy' % (num_images_to_avg, channel_index, downsize_factor, spatial_sigma_raw)

	if not os.path.exists(cache_filename):
		channel, _ = load_channel_progression_brightness_filtered(folder, bias_img, downsize_factor, channel_index, num_images_to_avg)

		print('spatial filtering channel')
		for ti in tqdm.tqdm(range(channel.shape[0])):
			channel[ti] = gaussian_filter(channel[ti], sigma = spatial_sigma_raw / downsize_factor)

		np.save(cache_filename, channel)

	else:
		print('loading from cache file...')
		start = datetime.datetime.now()
		channel = np.load(cache_filename)
		print('done', (datetime.datetime.now() - start))
		print(channel.shape)

	return channel, np.mean(channel, axis=(1,2))

def test_flatten_channel():
	# folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
	folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	bias_img = load_dark(bias_folder)
	downsize_factor = 4
	channel_index = 0

	test_img_path = 'F:/Pictures/Lightroom/2020/2020-02-18/orion_135mm/DSC03637.ARW'
	
	test_img = histogram_gap.load_raw_image(test_img_path, bias_img) 
	test_img_channel = test_img[:, :, channel_index]

	#todo: spatial and temporal filtering
	# channel_progression = load_channel_progression(folder, bias_img, downsize_factor, channel_index)
	# channel_progression, image_means = load_channel_progression_brightness_filtered(folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
	channel_progression, image_means = load_channel_progression_brightness_filtered_spatialfiltered(folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
	
	channel_flat = np.zeros_like(test_img_channel)

	print(test_img_channel.shape, channel_progression.shape, channel_flat.shape)

	for yi in tqdm.tqdm(range(test_img_channel.shape[0])):
		if 0:
			for xi in range(test_img_channel.shape[1]):
				input_pixel = test_img_channel[yi,xi]

				progression = channel_progression[:, yi//downsize_factor, xi//downsize_factor]

				index = np.argmin(np.abs(progression - input_pixel))
				
				if 0:
					print(progression.shape, index)
					plt.plot(np.arange(0, len(progression)), progression)
					plt.scatter([index], [input_pixel], 'r')
					plt.grid(True)
					plt.show()

				normalized_progression = progression / image_means
				output = normalized_progression[index] #* image_means[index]

				# output = image_means[index] /
				# output = progression[index] / (np.mean(progression) * image_means[index])
				# print(progression[index], np.mean(progression), image_means[index])
				# print(output)

				channel_flat[yi,xi] = output 
		else:
			# input_pixels = test_img_channel[yi]

			input_pixels = test_img_channel[yi]
			input_pixels = input_pixels.reshape((input_pixels.shape[0]//downsize_factor, downsize_factor)).mean(1)



			progressions = channel_progression[:, yi//downsize_factor, :]
			indices = np.argmin(np.abs(progressions - input_pixels), axis=0)
			# print('indices: ', indices.shape)
			# print('progressions: ', progressions.shape)
			# normalized_progressions = progressions / image_means
			# normalized_progressions = progressions.copy()
			# for i in range(normalized_progressions.shape[0]):
			# 	normalized_progressions[i] /= image_means[i]

			# outputs = normalized_progressions[:, indices]
			# print(outputs.shape, normalized_progressions.shape)

			outputs = np.zeros((len(indices)), dtype=channel_flat.dtype)
			for i in range(indices.shape[0]):
				# normalized_progression = progressions[:, i] / image_means
				# output = normalized_progression[indices[i]]

				output = progressions[indices[i], i] / image_means[indices[i]]
				outputs[i] = output

			channel_flat[yi] = np.repeat(outputs, downsize_factor)

			# print(channel_flat[yi, :10], outputs[:10])
			# exit(0)



	plt.imshow(channel_flat)
	plt.title('channel flat')
	plt.show()

	calibrated_img = test_img_channel / channel_flat

	plt.imshow(test_img_channel)
	plt.title('test img')
	plt.show()

	plt.imshow(calibrated_img)
	plt.title('calibrated test img')
	plt.show()

	# plt.hist(calibrated_img.flatten(), bins = 1000)
	# plt.show()

	benchmark_flat = full_flat_sequence.get_flat_matching_brightness_histogram_match(folder, test_img_channel, channel_index, bias_img = bias_img, half_images_to_average = 5)

	plt.imshow(benchmark_flat)
	plt.title('benchmark')
	plt.show()

	benchmark_ratio = channel_flat / benchmark_flat
	plt.title('difference v benchmark')
	plt.imshow(benchmark_ratio)
	plt.show()

def test1():
	# folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
	folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	bias_img = load_dark(bias_folder)
	downsize_factor = 4
	channel_index = 0

	channel = load_channel_progression(folder, bias_img, downsize_factor, channel_index)

	image_means = np.mean(channel, axis=(1,2))
	# print(image_means.shape)

	x = 1280
	y = 1000
	xs = []
	ys = []
	ys2 = []

	if 0:
		plt.imshow(channel[500, :, :])
		plt.show()

	if 0:
		plt.subplot(2, 1, 1)
		plt.plot(image_means)
		plt.grid(True)

		means_relative_diffs = image_means[:-1] / image_means[1:]

		plt.subplot(2, 1, 2)
		plt.plot(means_relative_diffs)
		plt.grid(True)
		plt.show()

	patch_size = 40

	xs = np.arange(0, channel.shape[0])

	ys1 = channel[:, (1000 - patch_size)//downsize_factor : (1000 + patch_size) // downsize_factor, (1280 - patch_size)//downsize_factor : (1280 + patch_size) // downsize_factor] 
	ys2 = channel[:, (100 - patch_size)//downsize_factor : (100 + patch_size) // downsize_factor, (100 - patch_size)//downsize_factor : (100 + patch_size) // downsize_factor]
	print(ys1.shape)

	ys1 = np.mean(ys1, axis=(1,2))
	ys2 = np.mean(ys2, axis=(1,2))

	ys1 /= image_means
	ys2 /= image_means

		# #TODO: better mean?
		# channel_mean = np.mean(channel_downsized)

		# img_ratio = channel_downsized / channel_mean

		# xs.append(i)
		# ys.append(img_ratio[y,x])
		# ys2.append(img_ratio[50,50])

	# ys = np.array(ys)
	# ys2 = np.array(ys2)

	for ys, label in [(ys1, '1280,1000'), (ys2, '100, 100')]:
		ys_normalized = ys / np.mean(ys)
		# plt.plot(xs, ys_normalized, '--')
		n_filt =5
		filt = np.ones(n_filt) / n_filt
		ys_filtered = scipy.signal.filtfilt(filt, [1], ys_normalized)

		plt.plot(xs, ys_filtered, label=label)			

	# plt.plot(xs, ys1 / np.mean(ys1))
	# plt.plot(xs, ys2 / np.mean(ys2))
	plt.legend()
	plt.grid(True)
	plt.show()

def process_row(input_pixels, progressions, channel_means):

	indices = np.argmin(np.abs(progressions - input_pixels), axis=0)

	outputs = np.zeros((len(indices)), dtype=np.float32)
	for i in range(indices.shape[0]):
		output = progressions[indices[i], i] / channel_means[indices[i]]
		outputs[i] = output

	return outputs

from numba import jit

@jit(nopython=True, parallel=True)
def process_row2(input_pixels, progressions, channel_means):
	indices = np.array([np.argmin(np.abs(progressions[:, xi] - input_pixels[xi])) for xi in range(len(input_pixels))])

	outputs = np.zeros((len(indices)), dtype=np.float32)
	for i in range(indices.shape[0]):
		output = progressions[indices[i], i] / channel_means[indices[i]]
		outputs[i] = output

	return outputs


all_channel_progression = None
all_channel_means = None

def load_all_channels(flats_progression_folder, bias_img, downsize_factor):
	global all_channel_progression
	global all_channel_means

	for channel_index in range(4):
		# channel_progression, channel_means = load_channel_progression_brightness_filtered(flats_progression_folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
		channel_progression, channel_means = load_channel_progression_brightness_filtered_spatialfiltered(flats_progression_folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
		if all_channel_progression is None:
			all_channel_progression = np.zeros((4,) + channel_progression.shape, dtype=channel_progression.dtype)
			all_channel_means = np.zeros((4,) + channel_means.shape, dtype=channel_means.dtype)

		all_channel_progression[channel_index] = channel_progression
		all_channel_means[channel_index] = channel_means

def get_subimg_matched_flat(flat_images_rgb, img_rgb, flats_progression_folder, bias_img, downsize_factor=4):
	#TODO: double flat matching with flat_images_rgb?
	# global all_channel_progression
	# global all_channel_means

	output_flat = np.zeros_like(img_rgb)

	for channel_index in range(img_rgb.shape[0]):
		# channel_progression, channel_means = load_channel_progression_brightness_filtered(flats_progression_folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
		channel_progression, channel_means = load_channel_progression_brightness_filtered_spatialfiltered(flats_progression_folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
		
		test_img_channel = img_rgb[channel_index]

		for yi in tqdm.tqdm(range(test_img_channel.shape[0])):
		
			input_pixels = test_img_channel[yi]
			input_pixels = input_pixels.reshape((input_pixels.shape[0]//downsize_factor, downsize_factor)).mean(1)

			progressions = channel_progression[:, yi//downsize_factor, :]
			
			outputs = process_row(input_pixels, progressions, channel_means)

			output_flat[channel_index, yi] = np.repeat(outputs, downsize_factor)

	return output_flat

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
def get_flat_and_subimg_matched_flat(flat_images_rgb, test_img_rgb, flats_progression_folder, bias_img = None):

	proportiontocut = 0.2
	global flat_image_means
	if flat_image_means is None:
		flat_image_means = calc_flat_image_means(flat_images_rgb, proportiontocut)

	daily_flat = np.zeros_like(test_img_rgb)
	for channel in range(test_img_rgb.shape[0]):
		# output_flat[channel] = 1; continue
		# print('starting channel ', channel)
		test_img_channel_mean = scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = proportiontocut)
		# print('test img mean: ', test_img_channel_mean)
		# print(flat_image_means[:, channel] - test_img_channel_mean)
		closest_flat_index = np.abs(flat_image_means[:, channel] - test_img_channel_mean).argmin()

		# print('closest flat index: ', closest_flat_index)

		daily_flat[channel] = flat_images_rgb[closest_flat_index, channel, :, :]

	daily_flat_progression_flat = get_subimg_matched_flat2(None, daily_flat, flats_progression_folder, bias_img)
	test_img_progression_flat = get_subimg_matched_flat2(None, test_img_rgb, flats_progression_folder, bias_img)

	output_flat = test_img_progression_flat * daily_flat / daily_flat_progression_flat

	return output_flat

def get_subimg_matched_flat2(flat_images_rgb, img_rgb, flats_progression_folder, bias_img, downsize_factor=4):
	#TODO: double flat matching with flat_images_rgb?
	global all_channel_progression
	global all_channel_means

	if all_channel_progression is None:
		load_all_channels(flats_progression_folder, bias_img, downsize_factor)

	output_flat = np.zeros_like(img_rgb)

	for channel_index in range(img_rgb.shape[0]):
		channel_progression = all_channel_progression[channel_index]
		channel_means = all_channel_means[channel_index]
		test_img_channel = img_rgb[channel_index]

		# for yi in tqdm.tqdm(range(test_img_channel.shape[0])):
		for yi in range(test_img_channel.shape[0]):
		
			input_pixels = test_img_channel[yi]
			input_pixels = input_pixels.reshape((input_pixels.shape[0]//downsize_factor, downsize_factor)).mean(1)

			progressions = channel_progression[:, yi//downsize_factor, :]


			outputs = process_row2(input_pixels, progressions, channel_means)


			output_flat[channel_index, yi] = np.repeat(outputs, downsize_factor)

	return output_flat

def test_get_flat():
	# folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
	folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	bias_img = load_dark(bias_folder)
	downsize_factor = 4

	test_img_path = 'F:/Pictures/Lightroom/2020/2020-02-18/orion_135mm/DSC03637.ARW'
	
	test_img = histogram_gap.load_raw_image(test_img_path, bias_img) 


	test_img = np.transpose(test_img, (2, 0, 1))

	flat = get_subimg_matched_flat(None, test_img, folder, bias_img)

	for channel_index in range(4):
		print('testing channel ', channel_index)
		channel_flat = flat[channel_index]
		test_img_channel = test_img[channel_index]

		plt.imshow(channel_flat)
		plt.title('channel flat')
		plt.show()

		calibrated_img = test_img_channel / channel_flat

		plt.imshow(test_img_channel)
		plt.title('test img')
		plt.show()

		plt.imshow(calibrated_img)
		plt.title('calibrated test img')
		plt.show()

		# plt.hist(calibrated_img.flatten(), bins = 1000)
		# plt.show()

		benchmark_flat = full_flat_sequence.get_flat_matching_brightness_histogram_match(folder, test_img_channel, channel_index, bias_img = bias_img, half_images_to_average = 5)

		plt.imshow(benchmark_flat)
		plt.title('benchmark')
		plt.show()

		benchmark_ratio = channel_flat / benchmark_flat
		plt.title('difference v benchmark')
		plt.imshow(benchmark_ratio)
		plt.show()

def test_get_flat_faster():
	folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
	bias_folder = 'K:/orion_135mm_bothnights/bias'
	bias_img = load_dark(bias_folder)
	downsize_factor = 4

	test_img_path = 'F:/Pictures/Lightroom/2020/2020-02-18/orion_135mm/DSC03637.ARW'
	
	test_img = histogram_gap.load_raw_image(test_img_path, bias_img) 


	test_img = np.transpose(test_img, (2, 0, 1))

	# resize = 4
	# test_img = test_img[0:1, :test_img.shape[1]//(resize*downsize_factor) * downsize_factor, :test_img.shape[2]//(resize*downsize_factor) * downsize_factor]
	# test_img = test_img[0:1]
	print('test_img: ', test_img.shape)

	# for channel_index in range(test_img.shape[0]):
	# 	channel_progression, channel_means = load_channel_progression_brightness_filtered(folder, bias_img, downsize_factor, channel_index, num_images_to_avg=5)
	
	load_all_channels(folder, bias_img, downsize_factor)

	start = datetime.datetime.now()
	flat_new = get_subimg_matched_flat2(None, test_img, folder, bias_img)
	print('new time: ', (datetime.datetime.now() - start))

	if 1:
		start = datetime.datetime.now()
		flat_reference = get_subimg_matched_flat(None, test_img, folder, bias_img)
		print('reference time: ', (datetime.datetime.now() - start))

		if np.all(flat_new == flat_reference):
			print('ALL GOOD!')

		else:
			print('***not equal***')
			avg_diff = np.mean(np.abs(flat_new - flat_reference) / flat_reference)
			print('average relative difference: ', avg_diff)
			ratio = (flat_new / flat_reference)[0]
			plt.imshow(ratio)
			plt.show()


if __name__ == "__main__":
	# test1()
	test_flatten_channel()
	# test_get_flat()
	# test_get_flat_faster()

	# import cProfile
	# cProfile.run('test_get_flat_faster()')
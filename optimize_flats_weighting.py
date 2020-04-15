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

def remove_gradient(img, quadratic=True):
	x = np.linspace(0, 1, img.shape[1])
	y = np.linspace(0, 1, img.shape[0])
	X, Y = np.meshgrid(x, y, copy=False)
	
	X = X.flatten()
	Y = Y.flatten()

	if quadratic:
		A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
	else:
		A = np.array([X*0+1, X, Y]).T
	

	B = img[:, :].flatten()

	coeff, r, rank, s = np.linalg.lstsq(A, B)
	# coeff[0] = 0

	fit = np.reshape(A.dot(coeff), img.shape[:2])
	print(coeff)
	print(A.shape, coeff.shape, fit.shape)

	# plt.imshow(fit)
	# plt.show()
	out_img = img / fit * np.mean(fit)

	if 0:
		z = 10
		plt.subplot(2, 1, 1)
		low = np.percentile(img, z)
		high = np.percentile(img, 100 - z)
		plt.imshow(np.clip(img, low, high))
		plt.subplot(2, 1, 2)
		low = np.percentile(out_img, z)
		high = np.percentile(out_img, 100 - z)
		plt.imshow(np.clip(out_img, low, high))
		plt.show()

	return out_img

def normalize_image_mean(img):

			# for img in range(flat_images_rgb.shape[1]):
			# 	flat_image_means[channel, img] = scipy.stats.trim_mean(flat_images_rgb[channel, img].flatten(), proportiontocut = proportiontocut)
	mean = scipy.stats.trim_mean(img.flatten(), proportiontocut = 0.2)

	normalized_img = img / mean

	return normalized_img

def get_optimized_flat(channel_flats, test_channel):

	raw_channel_flats = channel_flats.copy()

	test_channel = remove_gradient(test_channel)
	channel_flats = [remove_gradient(flat) for flat in channel_flats]

	test_channel = normalize_image_mean(test_channel)
	channel_flats = [normalize_image_mean(flat) for flat in channel_flats]

	channel_flats = np.array(channel_flats)

	print('means: ', np.mean(test_channel), [np.mean(f) for f in channel_flats])

	channel_flats_squared = channel_flats ** 2
	channel_flats_root = channel_flats ** 0.5
	constant_val = np.ones((1, channel_flats.shape[1], channel_flats.shape[2]))

	# channel_flats = np.concatenate((channel_flats, channel_flats_squared, channel_flats_root, constant_val))
	# channel_flats = np.concatenate((channel_flats, 
	# 			# channel_flats_root,
	# 			constant_val))

	# raw_channel_flats = np.concatenate((raw_channel_flats, constant_val))

	print(np.mean(channel_flats, axis=(1,2)), np.mean(test_channel))


	print(channel_flats.shape, test_channel.shape)

	#todo: pull in previous flattening, use to reject stars, other outliers

	flattened_flats = channel_flats.reshape((channel_flats.shape[0], channel_flats.shape[1] * channel_flats.shape[2]))
	print(flattened_flats.shape)
	flattened_test = test_channel.flatten()

	a = np.transpose(flattened_flats, (1, 0))
	b = flattened_test

	print(a.shape, b.shape)
	if 1:
		weights, residuals, rank, singular_values = np.linalg.lstsq(a, b)
		print('weights: ', weights)
		print('residuals: ', residuals)
	# else:



	weighted_flat = np.transpose(channel_flats, (1, 2, 0)).dot(weights)
	print(weighted_flat.shape, np.mean(weighted_flat))

	flattened_test_img = test_channel / weighted_flat
	flattened_test_img_flat = flattened_test_img.flatten()
	
	if 0:
		plt.subplot(2, 1, 1)
		plt.imshow(test_channel)
		plt.subplot(2, 1, 2)
		plt.imshow(flattened_test_img)
		plt.show()

	if 0:
		plt.hist(flattened_test_img.flatten(), bins = 1000)
		plt.grid(True)
		plt.yscale('log', nonposy='clip')
		plt.show()

	indices = np.arange(0, len(flattened_test_img_flat))

	for improve_iter in range(5):

		flattened_test_img_flat = flattened_test_img.flatten()

		num_sigmas = 1.5
		# mu = np.mean(flattened_test_img)
		# sigma = np.std(flattened_test_img)
		# print('mean, std: ', mu, sigma)

		mu = np.mean(flattened_test_img_flat[indices])
		sigma = np.std(flattened_test_img_flat[indices])
		print('mean, std: ', mu, sigma)


		flattened_test_img_flat = flattened_test_img.flatten()
		indices = np.where(((mu - num_sigmas * sigma) < flattened_test_img_flat) & (flattened_test_img_flat < (mu + num_sigmas * sigma)))[0]

		print('% of indices to keep: ', len(indices) / len(flattened_test_img_flat))


		a2 = a[indices]
		b2 = b[indices]


		weights, residuals, rank, singular_values = np.linalg.lstsq(a2, b2)
		print('weights2: ', weights)
		print('residuals2: ', residuals, residuals / len(indices))
		weighted_flat = np.transpose(channel_flats, (1, 2, 0)).dot(weights)
		raw_weighted_flat = np.transpose(raw_channel_flats, (1, 2, 0)).dot(weights)

		flattened_test_img = test_channel / weighted_flat

		if 1:

			disp_image = gaussian_filter(flattened_test_img, mode='nearest', sigma=5)

			if 1:
				z = 1
				low = np.percentile(disp_image, z)
				high = np.percentile(disp_image, 100 - z)
				disp_image = np.clip(disp_image, low, high)
			else:
				z = 10
				low = np.percentile(disp_image, z)
				high = np.percentile(disp_image, 100 - z)
				disp_image[np.where(disp_image < low)] = 1
				disp_image[np.where(disp_image > high)] = 1 


			plt.imshow(disp_image)
			plt.show()

	return raw_weighted_flat

def optimize_flats_to_test_img():

	if 1:
		# flat_filenames = ['K:/Orion_135mm/flats/flats1_gray.tif', 
		# 'K:/Orion_135mm/flats/flats2_gray.tif', 
		# 'K:/Orion_135mm/flats/flats3_gray.tif',
		# 'K:/Orion_135mm/flats/flats4_gray.tif',
		# 'K:/Orion_135mm/flats/flats5_gray.tif',
		# 'K:/Orion_135mm/flats/flats_30scloth.tif',
		# ]


		flat_filenames = ['K:/Orion_135mm/flats/flats1_mean.tif', 
			'K:/Orion_135mm/flats/flats2_mean.tif', 
			'K:/Orion_135mm/flats/flats3_mean.tif',
			'K:/Orion_135mm/flats/flats4_mean.tif',
			'K:/Orion_135mm/flats/flats5_mean.tif',
		]

		test_img_folder = 'K:/Orion_135mm/lights_tiffs'
		lights_output_folder = 'K:/Orion_135mm/calibrated_lights_tiffs_optimized_meanflat'
	else:
		flat_filenames = ['K:/Orion_600mm/stacked_flats/flats1.tif',
		'K:/Orion_600mm/stacked_flats/flats2.tif',
		'K:/Orion_600mm/stacked_flats/flats3.tif',
		'K:/Orion_600mm/stacked_flats/flats4.tif',
		'K:/Orion_600mm/stacked_flats/flats5.tif',
		]
		
		test_img_folder = 'K:/Orion_600mm/lights_tiff'
		lights_output_folder = 'K:/Orion_600mm/calibrated_lights_tiffs_optimized'


	filenames = list(filter(lambda s: s.endswith('.tif'), os.listdir(test_img_folder)))#[:5]
	print(filenames)
	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	all_indices = []
	for filename in filenames:
		test_img = load_gray_tiff(os.path.join(test_img_folder, filename))		
		test_img_rgb = extract_channel_image(test_img)

		optimized_flat_rgb = np.zeros_like(test_img_rgb)
		for channel in range(4):
			flat_channel = get_optimized_flat(flat_images_rgb[:, channel], test_img_rgb[channel])
			# print(np.mean(flat_channel), np.mean(test_img_rgb[channel]))
			# exit(0)
			optimized_flat_rgb[channel] = flat_channel

		flattened_test_img_rgb = test_img_rgb / optimized_flat_rgb
		flattened_test_img_gray = flatten_channel_image(flattened_test_img_rgb)

		tiff.imwrite('optimized_flat.tif', flattened_test_img_gray.astype('float32'))

		tiff.imwrite(os.path.join(lights_output_folder, filename), flattened_test_img_gray.astype('float32'))

		# exit(0)

def remove_image_outliers(rgb_input_image):
	output_image = np.zeros_like(rgb_input_image)

	for channel, input_img in enumerate(rgb_input_image):
		img = input_img.copy()


		for i in range(10):
			blurred_background = gaussian_filter(img, mode='nearest', sigma=30)
			background_sub = img - blurred_background

			background_mean = np.mean(background_sub)
			background_std = np.std(background_sub)

			print('mean, std: ', background_mean, background_std)

			sigma = 2
			outlier_indices = np.where(np.abs(background_sub - background_mean) > sigma * background_std)
			
			# background_sub[outlier_indices] = background_mean
			img[outlier_indices] = blurred_background[outlier_indices]

		output_image[channel] = img

		if 0:
			def display_image(i):
				disp_image = i.copy()
				z = 1
				low = np.percentile(disp_image, z)
				high = np.percentile(disp_image, 100 - z)
				disp_image = np.clip(disp_image, low, high)
				# disp_image = np.log(disp_image)
				plt.imshow(disp_image)



			plt.subplot(3, 1, 1)
			display_image(input_img)

			plt.subplot(3, 1, 2)
			display_image(img)

			plt.subplot(3, 1, 3)
			display_image(background_sub)

			plt.show()

	return output_image

def optimize_flats_to_stacked_img():

	if 0:
		flat_filenames = ['K:/Orion_135mm/flats/flats1_gray.tif', 
		'K:/Orion_135mm/flats/flats2_gray.tif', 
		'K:/Orion_135mm/flats/flats3_gray.tif',
		'K:/Orion_135mm/flats/flats4_gray.tif',
		'K:/Orion_135mm/flats/flats5_gray.tif',
		'K:/Orion_135mm/flats/flats_30scloth.tif',
		'K:/Orion_135mm/flats/untracked_sky_30s_integration.tif'
		]


		# flat_filenames = [
		# 'K:/Orion_135mm/flats/flats_30scloth.tif',
		# 'K:/Orion_135mm/flats/untracked_sky_30s_integration.tif'
		# ]

		test_stacked_image = 'K:/Orion_135mm/unregistered_integration.tif'

	else:

		flat_filenames = ['K:/600mm_flats/02-08-pleiades.tif', 
		'K:/600mm_flats/02-19-flame-horsehead.tif', 
		'K:/600mm_flats/02-19-trails.tif', 
		'K:/600mm_flats/02-20-pleiades.tif', 
		'K:/600mm_flats/02-29-orion.tif', 
		'K:/600mm_flats/03-11-trails.tif', 
		]

		test_stacked_image = 'K:/600mm_flats/test_images/02-19-flame-horsehead.tif'

	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = [extract_channel_image(img) for img in flat_images]

	flat_images_rgb = [remove_image_outliers(img) for img in flat_images_rgb]
	flat_images_rgb = np.array(flat_images_rgb)

	test_img = load_gray_tiff(test_stacked_image)		
	test_img_rgb = extract_channel_image(test_img)
	test_img_no_outliers = remove_image_outliers(test_img_rgb)

	optimized_flat_rgb = np.zeros_like(test_img_rgb)
	for channel in range(4):

		# test_channel = test_img_rgb[channel]
		test_channel = test_img_no_outliers[channel]

		flat_channel = get_optimized_flat(flat_images_rgb[:, channel], test_channel)
		# print(np.mean(flat_channel), np.mean(test_img_rgb[channel]))
		# exit(0)
		optimized_flat_rgb[channel] = flat_channel

	flat_optimized_flat = flatten_channel_image(optimized_flat_rgb)

	tiff.imwrite('optimized_flat_for_stack.tif', flat_optimized_flat.astype('float32'))

def nonlinear_optimize_weights():

	flat_filenames = [
	'K:/Orion_135mm/flats/flats_30scloth.tif',
	'K:/Orion_135mm/flats/untracked_sky_30s_integration.tif',
	'K:/Orion_135mm/darks_30s.tif',
	]

	test_stacked_image = 'K:/Orion_135mm/unregistered_integration.tif'

	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	test_img = load_gray_tiff(test_stacked_image)		
	test_img_rgb = extract_channel_image(test_img)

	optimized_flat_rgb = np.zeros_like(test_img_rgb)
	for channel in range(4):

		channel_flats = flat_images_rgb[:, channel]
		test_channel = test_img_rgb[channel]

		display_images = [(channel_flats[0], 'flat cloth'), 
					(channel_flats[1], 'untracked sky'),
					(test_channel, 'test channel'),
					# (channel_flats[0], 'flat cloth'),
					]

		for i in range(len(display_images)):
			plt.subplot(2, 2, i+1)
			disp_image, title = display_images[i]
			
			# disp_image -= np.mean(disp_image)

			disp_image = remove_gradient(disp_image)
			disp_image = normalize_image_mean(disp_image)


			disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

			z = 1
			low = np.percentile(disp_image, z)
			high = np.percentile(disp_image, 100 - z)
			disp_image = np.clip(disp_image, low, high)


			plt.imshow(disp_image)
			plt.title(title)

		plt.show()

def subtract_dark(light, dark):
	result = light - dark
	result = np.clip(result, 0, 1E10)
	return result

def brute_force_opt():

	flat_filenames = [
	'K:/Orion_135mm/flats/flats_30scloth.tif',
	'K:/Orion_135mm/flats/untracked_sky_30s_integration.tif',
	]

	dark_filename = 'K:/Orion_135mm/darks_30s.tif'

	test_stacked_image = 'K:/Orion_135mm/unregistered_integration.tif'

	dark_image = load_gray_tiff(dark_filename)

	flat_images = [subtract_dark(load_gray_tiff(fn), dark_image) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	test_img = subtract_dark(load_gray_tiff(test_stacked_image), dark_image)
	test_img_rgb = extract_channel_image(test_img)

	optimized_flat_rgb = np.zeros_like(test_img_rgb)
	for channel in range(4):

		flat0, flat1 = flat_images_rgb[:, channel]
		test_channel = test_img_rgb[channel]

		weights = np.linspace(0, 1, 11)

		for weight in weights:
			flat = weight * flat0 + (1 - weight) * flat1

			output_image = test_channel / flat

			output_image = remove_gradient(output_image, quadratic=True)

			disp_image = output_image
			disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

			z = 1
			low = np.percentile(disp_image, z)
			high = np.percentile(disp_image, 100 - z)
			disp_image = np.clip(disp_image, low, high)


			plt.imshow(disp_image)
			plt.title(str(weight))
			plt.show()

def remove_circular_gradient():
	dark_filename = 'K:/Orion_135mm/darks_30s.tif'
	# test_stacked_image = 'K:/Orion_135mm/unregistered_integration.tif'

	test_stacked_image = 'K:/Orion_135mm/flats/untracked_sky_30s_integration.tif',

	dark_image = load_gray_tiff(dark_filename)

	test_img = load_gray_tiff(test_stacked_image)
	test_img = subtract_dark(test_img, dark_image)
	test_img_rgb = extract_channel_image(test_img)

	for channel in range(4):
		test_channel = test_img_rgb[channel]

		test_channel = remove_gradient(test_channel, quadratic=True)

		if 0:
			plt.imshow(test_channel)
			plt.show()

		cy = test_channel.shape[0] // 2 #+ 155
		cx = test_channel.shape[1] // 2 #+ 100

		max_radius = int(np.ceil(np.sqrt(max(cx, test_channel.shape[1] - cx)**2 + max(cy, test_channel.shape[0] - cy)**2)))
		# print('max radius: ', max_radius)

		pixels_by_radius = [[] for _ in range(max_radius)]

		for y in tqdm.tqdm(range(test_channel.shape[0])):
			for x in range(test_channel.shape[1]):
				r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
				pixels_by_radius[r].append(test_channel[y,x])

		def radius_mean_func(l):
			l2 = scipy.stats.trimboth(l, proportiontocut = 0.2)
			return np.mean(l2), np.std(l2)

		# radius_mean_func = lambda x: scipy.stats.trim_mean(x, proportiontocut = 0.2)
		# radius_mean_func = np.mean

		pixel_means_by_radius, pixel_stds_by_radius = map(np.array, zip(*[radius_mean_func(a) for a in pixels_by_radius]))

		if 0:
			plt.plot(pixel_means_by_radius)
			plt.plot(pixel_means_by_radius + pixel_stds_by_radius)
			plt.plot(pixel_means_by_radius - pixel_stds_by_radius)
			
			plt.show()

		# fit_data = np.concatenate((pixel_means_by_radius[::-1] + pixel_means_by_radius))
		fit_data = pixel_means_by_radius

		radial_fit = scipy.ndimage.filters.gaussian_filter1d(fit_data, sigma = 10, mode='reflect')

		if 0:
			plt.plot(fit_data)
			plt.plot(radial_fit)
			plt.show()

		flat_image = np.zeros_like(test_channel)

		for y in tqdm.tqdm(range(test_channel.shape[0])):
			for x in range(test_channel.shape[1]):
				r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
				flat_image[y,x] = radial_fit[r]

		calibrated_img = test_channel / flat_image


		disp_image = calibrated_img.copy()
		z = 1
		low = np.percentile(disp_image, z)
		high = np.percentile(disp_image, 100 - z)
		disp_image = np.clip(disp_image, low, high)
		plt.imshow(disp_image)
		plt.show()

def find_gradient_center():
	dark_filename = 'K:/Orion_135mm/darks_30s.tif'
	test_stacked_image = 'K:/Orion_135mm/unregistered_integration.tif'

	dark_image = load_gray_tiff(dark_filename)

	test_img = subtract_dark(load_gray_tiff(test_stacked_image), dark_image)
	test_img_rgb = extract_channel_image(test_img)

	for channel in range(4):
		test_channel = test_img_rgb[channel]

		c0y = test_channel.shape[0] // 2
		c0x = test_channel.shape[1] // 2

		c0y += 155
		c0x += 100

		# cy = c0y + 60
		# cx = c0x + 60

		z = 50
		n = 5
		cys = np.linspace(c0y - z, c0y + z, n)
		cxs = np.linspace(c0x - z, c0x + z, n)

		all_fit_qualities = np.zeros((n, n))

		for iy, cy in enumerate(cys):
			for ix, cx in enumerate(cxs):

				print(iy, ix)

				max_radius = int(np.ceil(np.sqrt(max(cx, test_channel.shape[1] - cx)**2 + max(cy, test_channel.shape[0] - cy)**2)))
				# print('max radius: ', max_radius)

				pixels_by_radius = [[] for _ in range(max_radius)]

				for y in tqdm.tqdm(range(test_channel.shape[0])):
					for x in range(test_channel.shape[1]):
						r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
						pixels_by_radius[r].append(test_channel[y,x])

				def radius_mean_func(l):
					l2 = scipy.stats.trimboth(l, proportiontocut = 0.2)
					return np.mean(l2), np.std(l2)

				# radius_mean_func = lambda x: scipy.stats.trim_mean(x, proportiontocut = 0.2)
				# radius_mean_func = np.mean

				pixel_means_by_radius, pixel_stds_by_radius = map(np.array, zip(*[radius_mean_func(a) for a in pixels_by_radius]))
				fit_quality = np.nanmean(pixel_stds_by_radius)
				# print(pixel_stds_by_radius)
				print(fit_quality)
				all_fit_qualities[iy, ix] = fit_quality
				if 0:
					plt.plot(pixel_means_by_radius)
					plt.plot(pixel_means_by_radius + pixel_stds_by_radius)
					plt.plot(pixel_means_by_radius - pixel_stds_by_radius)
					
					plt.show()

		print(all_fit_qualities)



if __name__ == "__main__":
	optimize_flats_to_test_img()

	# optimize_flats_to_stacked_img()
	# nonlinear_optimize_weights()
	# brute_force_opt()

	# find_gradient_center()
	# remove_circular_gradient()



	# test_stacked_image = 'K:/600mm_flats/test_images/02-19-flame-horsehead.tif'
	# test_img = load_gray_tiff(test_stacked_image)		
	# test_img_rgb = extract_channel_image(test_img)
	# remove_image_outliers(test_img_rgb)
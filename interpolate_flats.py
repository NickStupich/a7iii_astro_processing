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

def load_gray_tiff(fn):
	img = tiff.imread(fn)
	if not img.dtype in [np.float32, np.float64]:
		print('needs float img type!', img.dtype)
		print(fn)
		exit(0)

	# print(img.shape, img.dtype, np.mean(img))
	return img

def extract_channel_image(img):
	r = img[::2, ::2]
	g1 = img[1::2, ::2]
	g2 = img[::2, 1::2]
	b = img[1::2, 1::2]

	result = np.array([r, g1, g2, b])
	return result

def flatten_channel_image(img):
	result = np.zeros((2*img.shape[1], 2*img.shape[2]))
	result[::2, ::2] = img[0]
	result[1::2, ::2] = img[2]
	result[::2, 1::2] = img[1]
	result[1::2, 1::2] = img[3]

	return result

def get_correlation_matched_flat(flat_images_rgb, test_img_rgb):

	proportiontocut = 0.2

	flat_image_means = np.mean(flat_images_rgb, axis=(2, 3))
	print('flat image means: ', flat_image_means)

	if 1:
		for channel in range(flat_images_rgb.shape[0]):
			for img in range(flat_images_rgb.shape[1]):
				flat_image_means[channel, img] = scipy.stats.trim_mean(flat_images_rgb[channel, img].flatten(), proportiontocut = proportiontocut)

		# print('flat image means: ', flat_image_means)
	
	# test_img_means = np.mean(test_img_rgb, axis=(1,2))
	# print('test image means: ', test_img_means)

	calibrated_flat_img_rgb = np.zeros_like(flat_images_rgb[0])
	flat_image_weights = np.zeros((test_img_rgb.shape[0], len(flat_images_rgb)), dtype='float')
	interpolated_indices = []
	for channel in range(test_img_rgb.shape[0]):
		flat_image_channel_means = flat_image_means[:, channel]

		indices = np.arange(0, len(flat_image_channel_means))

		if 0:
			proportions = np.linspace(0, 0.3, 20)
			clipped_means = [scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = p) for p in proportions]
			plt.plot(proportions, clipped_means)
			plt.grid(True)
			plt.show()

		clipped_mean = scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = proportiontocut)
		test_img_channel_mean = clipped_mean
		# print('clipped mean: ', clipped_mean)

		for mean_iteration in range(0):

			if 0:
				plt.hist(test_img_rgb[channel].flatten(), bins = 1000)
				plt.yscale('log', nonposy='clip')
				plt.grid(True)
				plt.title(str(test_img_channel_mean))
				plt.show()

			interpolated_index = np.interp(test_img_channel_mean, flat_image_channel_means, indices)
			interpolated_indices.append(interpolated_index)
			print('interpolatioin: ', test_img_channel_mean, flat_image_channel_means, indices, interpolated_index)

			#TODO: bug on high end if flats don't go bright enough?
			flat_image_weights[channel, int(np.ceil(interpolated_index))] = (interpolated_index - int(np.floor(interpolated_index)))
			flat_image_weights[channel, int(np.floor(interpolated_index))] = 1 - (interpolated_index - int(np.floor(interpolated_index)))

			calibrated_flat_img_rgb[channel] = 0
			for flat_index in range(flat_image_weights.shape[1]):
				calibrated_flat_img_rgb[channel] += flat_image_weights[channel, flat_index] * flat_images_rgb[flat_index, channel]

				ratios = (test_img_rgb[channel] / calibrated_flat_img_rgb[channel]).flatten()
				n, bins, patches = plt.hist(ratios, bins = np.linspace(0.98, 1.02, 1001))
				bins = (bins[1:] + bins[:-1])/2
				fit = np.polyfit(bins, n, deg=2)
				peak = -fit[1] / (2*fit[0])
				
			if 1:
				plt.subplot(3, 1, 1)
				plt.imshow(np.log(calibrated_flat_img_rgb[channel]))

				plt.subplot(3, 1, 2)
				plt.imshow(np.log(test_img_rgb[channel]))

				plt.subplot(3, 1, 3)
				# plt.hist(ratios, bins = 1000)
				plt.title('peak: %.4f' % peak)
				plt.plot(bins, np.polyval(fit, bins), c = 'r')

				plt.yscale('log', nonposy='clip')
				plt.grid(True)

				plt.show()

			# peak = 1
			print('peak: ', peak)
			print('channel mean change: ', test_img_channel_mean, test_img_channel_mean * peak)
			test_img_channel_mean *= peak



		#use prev starting point, calculate correlations between light and flats on inliers only
		flat_image_correlations = np.zeros((len(flat_images_rgb)), dtype='float')
		for img in range(flat_images_rgb.shape[1]):
			flat_image_correlations[img] = scipy.stats.trim_mean(flat_images_rgb[channel, img].flatten(), proportiontocut = proportiontocut)



		# print(flat_image_weights[channel])

	#todo: in one step?
	# print(flat_images_rgb.shape, flat_image_weights.shape)
	# calibrated_flat_img = flat_image_weights.dot(flat_images_rgb)
	# print(calibrated_flat_img.shape)

	calibrated_flat_img_rgb = np.zeros_like(flat_images_rgb[0])
	for channel in range(flat_image_weights.shape[0]):
		for flat_index in range(flat_image_weights.shape[1]):
			calibrated_flat_img_rgb[channel] += flat_image_weights[channel, flat_index] * flat_images_rgb[flat_index, channel]

	# flat_template_weight_index = 2
	# for channel in range(calibrated_flat_img_rgb.shape[0]):
	# 	calibrated_flat_img_rgb[channel] *= np.mean(flat_images_rgb[flat_template_weight_index, channel]) / np.mean(calibrated_flat_img_rgb[channel])

	print('calibrated_flat_img: ', np.mean(calibrated_flat_img_rgb, axis=(1, 2)))

	if 0:
		for channel in range(4):
			plt.imshow(calibrated_flat_img_rgb[channel])
			plt.show()

	if 0:
		for channel in range(test_img_rgb.shape[0]):
			plt.subplot(3, 1, 1)
			plt.imshow(np.log(calibrated_flat_img_rgb[channel]))

			plt.subplot(3, 1, 2)
			plt.imshow(np.log(test_img_rgb[channel]))

			plt.subplot(3, 1, 3)
			ratios = (test_img_rgb[channel] / calibrated_flat_img_rgb[channel]).flatten()
			# plt.hist(ratios, bins = 1000)
			n, bins, patches = plt.hist(ratios, bins = np.linspace(0.98, 1.02, 1001))
			bins = (bins[1:] + bins[:-1])/2
			fit = np.polyfit(bins, n, deg=2)
			peak = -fit[1] / (2*fit[0])
			plt.title('peak: %.4f' % peak)
			plt.plot(bins, np.polyval(fit, bins), c = 'r')

			plt.yscale('log', nonposy='clip')
			plt.grid(True)

			plt.show()
	# exit(0)

	return calibrated_flat_img_rgb, interpolated_indices


def get_exposure_matched_flat(flat_images_rgb, test_img_rgb):

	proportiontocut = 0.2

	flat_image_means = np.mean(flat_images_rgb, axis=(2, 3))
	print('flat image means: ', flat_image_means)

	if 1:
		for channel in range(flat_images_rgb.shape[0]):
			for img in range(flat_images_rgb.shape[1]):
				flat_image_means[channel, img] = scipy.stats.trim_mean(flat_images_rgb[channel, img].flatten(), proportiontocut = proportiontocut)

		# print('flat image means: ', flat_image_means)
	
	# test_img_means = np.mean(test_img_rgb, axis=(1,2))
	# print('test image means: ', test_img_means)

	calibrated_flat_img_rgb = np.zeros_like(flat_images_rgb[0])
	flat_image_weights = np.zeros((test_img_rgb.shape[0], len(flat_images_rgb)), dtype='float')
	interpolated_indices = []
	for channel in range(test_img_rgb.shape[0]):
		flat_image_channel_means = flat_image_means[:, channel]

		indices = np.arange(0, len(flat_image_channel_means))

		if 0:
			proportions = np.linspace(0, 0.3, 20)
			clipped_means = [scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = p) for p in proportions]
			plt.plot(proportions, clipped_means)
			plt.grid(True)
			plt.show()

		clipped_mean = scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = proportiontocut)
		test_img_channel_mean = clipped_mean
		# print('clipped mean: ', clipped_mean)

		for mean_iteration in range(3):

			if 0:
				plt.hist(test_img_rgb[channel].flatten(), bins = 1000)
				plt.yscale('log', nonposy='clip')
				plt.grid(True)
				plt.title(str(test_img_channel_mean))
				plt.show()

			interpolated_index = np.interp(test_img_channel_mean, flat_image_channel_means, indices)
			interpolated_indices.append(interpolated_index)
			print('interpolatioin: ', test_img_channel_mean, flat_image_channel_means, indices, interpolated_index)

			#TODO: bug on high end if flats don't go bright enough?
			flat_image_weights[channel, int(np.ceil(interpolated_index))] = (interpolated_index - int(np.floor(interpolated_index)))
			flat_image_weights[channel, int(np.floor(interpolated_index))] = 1 - (interpolated_index - int(np.floor(interpolated_index)))

			calibrated_flat_img_rgb[channel] = 0
			for flat_index in range(flat_image_weights.shape[1]):
				calibrated_flat_img_rgb[channel] += flat_image_weights[channel, flat_index] * flat_images_rgb[flat_index, channel]

				ratios = (test_img_rgb[channel] / calibrated_flat_img_rgb[channel]).flatten()
				n, bins, patches = plt.hist(ratios, bins = np.linspace(0.98, 1.02, 1001))
				bins = (bins[1:] + bins[:-1])/2
				fit = np.polyfit(bins, n, deg=2)
				peak = -fit[1] / (2*fit[0])
				
			if 0:
				plt.subplot(3, 1, 1)
				plt.imshow(np.log(calibrated_flat_img_rgb[channel]))

				plt.subplot(3, 1, 2)
				plt.imshow(np.log(test_img_rgb[channel]))

				plt.subplot(3, 1, 3)
				# plt.hist(ratios, bins = 1000)
				plt.title('peak: %.4f' % peak)
				plt.plot(bins, np.polyval(fit, bins), c = 'r')

				plt.yscale('log', nonposy='clip')
				plt.grid(True)

				plt.show()

			# peak = 1
			print('peak: ', peak)
			print('channel mean change: ', test_img_channel_mean, test_img_channel_mean * peak)
			test_img_channel_mean *= peak

		# print(flat_image_weights[channel])

	#todo: in one step?
	# print(flat_images_rgb.shape, flat_image_weights.shape)
	# calibrated_flat_img = flat_image_weights.dot(flat_images_rgb)
	# print(calibrated_flat_img.shape)

	calibrated_flat_img_rgb = np.zeros_like(flat_images_rgb[0])
	for channel in range(flat_image_weights.shape[0]):
		for flat_index in range(flat_image_weights.shape[1]):
			calibrated_flat_img_rgb[channel] += flat_image_weights[channel, flat_index] * flat_images_rgb[flat_index, channel]

	# flat_template_weight_index = 2
	# for channel in range(calibrated_flat_img_rgb.shape[0]):
	# 	calibrated_flat_img_rgb[channel] *= np.mean(flat_images_rgb[flat_template_weight_index, channel]) / np.mean(calibrated_flat_img_rgb[channel])

	print('calibrated_flat_img: ', np.mean(calibrated_flat_img_rgb, axis=(1, 2)))

	if 0:
		for channel in range(4):
			plt.imshow(calibrated_flat_img_rgb[channel])
			plt.show()

	if 0:
		for channel in range(test_img_rgb.shape[0]):
			plt.subplot(3, 1, 1)
			plt.imshow(np.log(calibrated_flat_img_rgb[channel]))

			plt.subplot(3, 1, 2)
			plt.imshow(np.log(test_img_rgb[channel]))

			plt.subplot(3, 1, 3)
			ratios = (test_img_rgb[channel] / calibrated_flat_img_rgb[channel]).flatten()
			# plt.hist(ratios, bins = 1000)
			n, bins, patches = plt.hist(ratios, bins = np.linspace(0.98, 1.02, 1001))
			bins = (bins[1:] + bins[:-1])/2
			fit = np.polyfit(bins, n, deg=2)
			peak = -fit[1] / (2*fit[0])
			plt.title('peak: %.4f' % peak)
			plt.plot(bins, np.polyval(fit, bins), c = 'r')

			plt.yscale('log', nonposy='clip')
			plt.grid(True)

			plt.show()
	# exit(0)

	return calibrated_flat_img_rgb, interpolated_indices

def exposures_throughout_night():
	if 1:
		# flat_filenames = ['K:/orion_135mm/flats/flats1_gray.tif', 
		# 'K:/orion_135mm/flats/flats2_gray.tif', 
		# 'K:/orion_135mm/flats/flats3_gray.tif',
		# 'K:/orion_135mm/flats/flats4_gray.tif',
		# ]

		flat_filenames = ['K:/Orion_135mm/flats/flats1_mean.tif', 
			'K:/Orion_135mm/flats/flats2_mean.tif', 
			'K:/Orion_135mm/flats/flats3_mean.tif',
			'K:/Orion_135mm/flats/flats4_mean.tif',
			'K:/Orion_135mm/flats/flats5_mean.tif',
		]


		test_img_folder = 'K:/orion_135mm/lights_tiffs'
		# lights_output_folder = 'K:/orion_135mm/calibrated_lights_tiffs'
		lights_output_folder = 'K:/Orion_135mm/calibrated_lights_tiffs_optimized_meanflat'

	elif 0:

		flat_filenames = ['F:/pixinsight_learning/pleiades/gray_flats/flats5.tif', 
		'F:/pixinsight_learning/pleiades/gray_flats/flats4.tif', 
		'F:/pixinsight_learning/pleiades/gray_flats/flats3.tif',
		'F:/pixinsight_learning/pleiades/gray_flats/flats2.tif',
		'F:/pixinsight_learning/pleiades/gray_flats/flats1.tif'
		]

		test_img_folder = 'F:/pixinsight_learning/pleiades/pleiades_gray_tiffs'
		lights_output_folder = 'F:/pixinsight_learning/pleiades/pleiades_calibrated_tiffs'

	elif 0:
		flat_filenames = ['K:/Orion_600mm/stacked_flats/flats1.tif',
			'K:/Orion_600mm/stacked_flats/flats2.tif',
			'K:/Orion_600mm/stacked_flats/flats3.tif',
			'K:/Orion_600mm/stacked_flats/flats4.tif',
			'K:/Orion_600mm/stacked_flats/flats5.tif',
			]

		test_img_folder = 'K:/Orion_600mm/lights_tiff'
		lights_output_folder = 'K:/Orion_600mm/calibrated_lights_tiff'
		interpolated_flats_folder = 'K:/Orion_600mm/interpolated_flats'
	elif 0:

		flat_filenames = ['K:/Orion_135mm/flats/flats1_gray.tif', 
		'K:/Orion_135mm/flats/flats2_gray.tif', 
		'K:/Orion_135mm/flats/flats3_gray.tif',
		'K:/Orion_135mm/flats/flats4_gray.tif',
		]

		test_img_folder = 'K:/Orion_135mm/lights_tiffs'
		lights_output_folder = 'K:/Orion_135mm/calibrated_lights_tiffs'
		interpolated_flats_folder = 'K:/Orion_135mm/interpolated_flats'
	else:
		flat_filenames = ['K:/Orion_135mm/flats/flats1_gray.tif', 
		'K:/Orion_135mm/flats/flats2_gray.tif', 
		'K:/Orion_135mm/flats/flats3_gray.tif',
		'K:/Orion_135mm/flats/flats4_gray.tif',
		]

		test_img_folder = 'K:/Orion_135mm/lights_tiffs'
		lights_output_folder = 'K:/Orion_135mm/calibrated_lights_tiffs'
		interpolated_flats_folder = 'K:/Orion_135mm/interpolated_flats'
	
	filenames = list(filter(lambda s: s.endswith('.tif'), os.listdir(test_img_folder)))#[:5]
	print(filenames)
	
	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])


	all_indices = []
	for filename in filenames:
		test_img = load_gray_tiff(os.path.join(test_img_folder, filename))
		test_img_rgb = extract_channel_image(test_img)

		calibrated_flat_img_rgb, exposure_index = get_exposure_matched_flat(flat_images_rgb, test_img_rgb)
		# calibrated_flat_img_rgb, exposure_index = get_correlation_matched_flat(flat_images_rgb, test_img_rgb)

		# print('now compare?')
		# exit(0)

		calibrated_flat_nonnormalized = flatten_channel_image(calibrated_flat_img_rgb)

		# for channel in range(calibrated_flat_img_rgb.shape[0]):
		# 	calibrated_flat_img_rgb[channel] /= np.mean(calibrated_flat_img_rgb[channel])
		# s0 = np.mean(calibrated_flat_img_rgb)
		# print('s0: ', s0)
		# # calibrated_flat_img_rgb /= s0

		calibrated_flat_img = flatten_channel_image(calibrated_flat_img_rgb)

		calibrated_test_img = test_img / calibrated_flat_img

		calibrated_test_img_rgb = extract_channel_image(calibrated_test_img)
		
		if 0:
			plt.subplot(3, 1, 1)		
			for channel, color in zip(test_img_rgb, ['r', 'g', 'g', 'b']):
				plt.hist(channel.flatten(), bins = 100, color=color, histtype='step')
			plt.grid(True)
			plt.title('input image')
			plt.yscale('log', nonposy='clip')

			plt.subplot(3, 1, 2)		
			for channel, color in zip(calibrated_flat_img_rgb, ['r', 'g', 'g', 'b']):
				plt.hist(channel.flatten(), bins = 100, color=color, histtype='step')
			plt.grid(True)
			plt.title('calibrated flat image')
			plt.yscale('log', nonposy='clip')

			plt.subplot(3, 1, 3)
			for channel, color in zip(calibrated_test_img_rgb, ['r', 'g', 'g', 'b']):
				plt.hist(channel.flatten(), bins = 100, color=color, histtype='step')
			plt.grid(True)
			plt.yscale('log', nonposy='clip')
			plt.title('calibrated image')
			plt.show()

		if 0:
			for channel in range(calibrated_test_img_rgb.shape[0]):	
				img_data = calibrated_test_img_rgb[channel]	
				c = 5
				clip_low, clip_high = np.percentile(img_data.flatten(), [c, 100 - c])
				img_data = np.clip(img_data, clip_low, clip_high)
				img_data = np.log(img_data)
				plt.imshow(img_data)
				plt.show()

		print(np.mean(calibrated_test_img))


		tiff.imwrite(os.path.join(lights_output_folder, filename), calibrated_test_img.astype('float32'))

		# tiff.imwrite(os.path.join(interpolated_flats_folder, filename), calibrated_flat_nonnormalized.astype('float32'))

		all_indices.append(exposure_index)

	all_indices = np.array(all_indices)
	print(all_indices.shape)

	for i, c in enumerate(['r', 'g', 'g', 'b']):
		plt.plot(all_indices[:, i], color=c)

	plt.grid(True)
	plt.show()

def flats_subtraction():


	flat_filenames = ['K:/Orion_135mm/flats/flats1_gray.tif', 
	'K:/Orion_135mm/flats/flats2_gray.tif', 
	'K:/Orion_135mm/flats/flats3_gray.tif',
	'K:/Orion_135mm/flats/flats4_gray.tif',
	]

	test_img_folder = 'K:/Orion_135mm/lights_tiffs'


	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	filenames = list(filter(lambda s: s.endswith('.tif'), os.listdir(test_img_folder)))#[:5]
	print(filenames)

	img0 = flat_images_rgb[2]
	img1 = flat_images_rgb[3]

	print('image means: ', np.mean(img0), np.mean(img1))

	bias_constant = 0.0001

	img0 -= bias_constant
	img1 -= bias_constant

	proportiontocut = 0.2

	for channel in range(img0.shape[0]):
		channel0 = img0[channel]
		channel1 = img1[channel]
		clipped_mean0 = scipy.stats.trim_mean(channel0.flatten(), proportiontocut = proportiontocut)
		clipped_mean1 = scipy.stats.trim_mean(channel1.flatten(), proportiontocut = proportiontocut)

		ratio = clipped_mean1 / clipped_mean0
		print(clipped_mean0, clipped_mean1, ratio)

		ratio2 = scipy.stats.trim_mean((channel1 / channel0).flatten(), proportiontocut = proportiontocut)
		print('ratio2: ', ratio2)

		scaled_img0 = channel0 * ratio2
		difference_img = clipped_mean1 / scaled_img0

		print('difference std: ', np.std(difference_img.flatten()))
		plt.imshow(difference_img)
		plt.show()


		# exit(0)
	

	# all_indices = []
	# for filename in filenames:
	# 	test_img = load_gray_tiff(os.path.join(test_img_folder, filename))

		
	# 	test_img_rgb = extract_channel_image(test_img)

def flats_linearity():


	flat_filenames = ['K:/Orion_135mm/flats/flats1_gray.tif', 
	'K:/Orion_135mm/flats/flats2_gray.tif', 
	'K:/Orion_135mm/flats/flats3_gray.tif',
	'K:/Orion_135mm/flats/flats4_gray.tif',
	]

	test_img_folder = 'K:/Orion_135mm/lights_tiffs'


	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	for channel in range(4):
		channel_flats = flat_images_rgb[:, channel, :, :]
		print(channel_flats.shape)

		# plt.imshow(channel_flats[0])
		# plt.show()


		img_means = np.mean(channel_flats, axis=(1,2))
		print(img_means)
		plt.plot(img_means)
		plt.show()


def main():
	flat_filenames = ['F:/pixinsight_learning/orion_135mm/flats/flats1_gray.tif', 
	'F:/pixinsight_learning/orion_135mm/flats/flats2_gray.tif', 
	'F:/pixinsight_learning/orion_135mm/flats/flats3_gray.tif',
	'F:/pixinsight_learning/orion_135mm/flats/flats4_gray.tif',
	]

	output_flat_filename = 'F:/pixinsight_learning/orion_135mm/flats/img_calibrated_flat.tif'

	test_img_filename = ['F:/pixinsight_learning/orion_135mm/test_images/DSC03694.tif']

	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	test_img = load_gray_tiff(test_img_filename)

	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])
	
	test_img_rgb = extract_channel_image(test_img)

	calibrated_flat_img_rgb, exposure_index = get_exposure_matched_flat(flat_images_rgb, test_img_rgb)

	calibrated_flat_img = flatten_channel_image(calibrated_flat_img_rgb)
	
	# plt.imshow(calibrated_flat_img)
	# plt.show()

	tiff.imwrite(output_flat_filename, calibrated_flat_img)

if __name__ == "__main__":
	# main()
	exposures_throughout_night()
	# flats_subtraction()
	# flats_linearity()
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
import datetime

from interpolate_flats import load_gray_tiff, extract_channel_image, flatten_channel_image

def remove_gradient(img, quadratic=False):
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

	# print(A, B, coeff, r, rank, s)
	# coeff[0] = 0

	fit = np.reshape(A.dot(coeff), img.shape[:2])
	# print(coeff)
	# print(A.shape, coeff.shape, fit.shape)

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

def display_image(i, z=1):
	disp_image = i.copy()

	disp_image = remove_gradient(disp_image)

	disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

	low = np.percentile(disp_image, z)
	high = np.percentile(disp_image, 100 - z)
	disp_image = np.clip(disp_image, low, high)
	plt.imshow(disp_image)

def get_exposure_matched_flat(flat_images_rgb, test_img_rgb):

	proportiontocut = 0.2

	flat_image_means = np.mean(flat_images_rgb, axis=(2, 3))
	print('flat image means: ', flat_image_means)

	if 1:
		for channel in range(flat_images_rgb.shape[0]):
			for img in range(flat_images_rgb.shape[1]):
				flat_image_means[channel, img] = scipy.stats.trim_mean(flat_images_rgb[channel, img].flatten(), proportiontocut = proportiontocut)

		print('flat image means: ', flat_image_means)
	
	# test_img_means = np.mean(test_img_rgb, axis=(1,2))
	# print('test image means: ', test_img_means)

	calibrated_flat_img_rgb = np.zeros_like(flat_images_rgb[0])
	flat_image_weights = np.zeros((test_img_rgb.shape[0], len(flat_images_rgb)), dtype='float')
	interpolated_indices = []
	for channel in range(test_img_rgb.shape[0]):
		print('starting channel ', channel)
		flat_image_channel_means = flat_image_means[:, channel]

		indices = np.arange(0, len(flat_image_channel_means))

		clipped_mean = scipy.stats.trim_mean(test_img_rgb[channel].flatten(), proportiontocut = proportiontocut)
		test_img_channel_mean = clipped_mean

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
			flat_image_weights[channel] = 0
			flat_image_weights[channel, int(np.ceil(interpolated_index))] = (interpolated_index - int(np.floor(interpolated_index)))
			flat_image_weights[channel, int(np.floor(interpolated_index))] = 1 - (interpolated_index - int(np.floor(interpolated_index)))
			print('flat image weights: ', flat_image_weights[channel])

			calibrated_flat_img_rgb[channel] = 0
			for flat_index in range(flat_image_weights.shape[1]):
				calibrated_flat_img_rgb[channel] += flat_image_weights[channel, flat_index] * flat_images_rgb[flat_index, channel]

			for peak_range in [0.02, 0.05, 0.1, 0.2]:
				all_ratios = (test_img_rgb[channel] / calibrated_flat_img_rgb[channel]).flatten()
				ratios = all_ratios[np.where(np.abs(all_ratios - 1) < peak_range)]
				n, bins = np.histogram(ratios, bins = np.linspace(1 - peak_range, 1 + peak_range, 1001))

				bins = (bins[1:] + bins[:-1])/2
				fit = np.polyfit(bins, n, deg=2)
				peak = -fit[1] / (2*fit[0])

				if np.abs(peak - 1) < peak_range:
					break

			if 0:
				flattened_channel = test_img_rgb[channel] / calibrated_flat_img_rgb[channel]
				display_image(flattened_channel, z = 10)
				plt.show()

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


			if np.isnan(peak):
				return None, None
				
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

def test():
	flat_filenames = ['K:/orion_135mm_histfix/integrated_flats/1.tif', 
	'K:/orion_135mm_histfix/integrated_flats/2.tif', 
	'K:/orion_135mm_histfix/integrated_flats/3.tif', 
	'K:/orion_135mm_histfix/integrated_flats/4.tif', 
	'K:/orion_135mm_histfix/integrated_flats/5.tif', 
	]

	test_filename = 'K:/orion_135mm_histfix/lights_with_dark/DSC03677_histfix.tif'
	


	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	test_img = load_gray_tiff(test_filename)
	test_img_rgb = extract_channel_image(test_img)

	# for channel in range(4):

	before = datetime.datetime.now()
	get_exposure_matched_flat(flat_images_rgb, test_img_rgb)
	after = datetime.datetime.now()
	print('elapsed time: ', (after - before))

def process_folder():
	flat_filenames = ['K:/orion_135mm_histfix/integrated_flats/1.tif', 
	'K:/orion_135mm_histfix/integrated_flats/2.tif', 
	'K:/orion_135mm_histfix/integrated_flats/3.tif', 
	'K:/orion_135mm_histfix/integrated_flats/4.tif', 
	'K:/orion_135mm_histfix/integrated_flats/5.tif', 
	]

	test_img_folder = 'K:/orion_135mm_histfix/lights_with_dark_interpolated_cal'
	lights_output_folder = 'K:/orion_135mm_histfix/interpolated_flats'
	

	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	filenames = list(filter(lambda s: s.endswith('.tif'), os.listdir(test_img_folder)))#[:5]
	print(filenames)
	
	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])


	all_indices = []
	for filename in filenames:
		test_img = load_gray_tiff(os.path.join(test_img_folder, filename))
		test_img_rgb = extract_channel_image(test_img)

		calibrated_flat_img_rgb, exposure_index = get_exposure_matched_flat(flat_images_rgb, test_img_rgb)

		if calibrated_flat_img_rgb is None: 
			print("***Failed to find matching flat")
			continue

		calibrated_flat_img = flatten_channel_image(calibrated_flat_img_rgb)
		calibrated_test_img = test_img / calibrated_flat_img
		calibrated_test_img_rgb = extract_channel_image(calibrated_test_img)
		
		tiff.imwrite(os.path.join(lights_output_folder, filename), calibrated_test_img.astype('float32'))

		all_indices.append(exposure_index)

	all_indices = np.array(all_indices)
	print(all_indices.shape)

	for i, c in enumerate(['r', 'g', 'g', 'b']):
		plt.plot(all_indices[:, i], color=c)

	plt.grid(True)
	plt.show()

if __name__ == "__main__":
	test()
	# process_folder()
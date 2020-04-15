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

def load_raw_image(path):

	if 0:
		with rawpy.imread(path) as raw:

			# raw_colors = np.array(raw.raw_colors)
			# print(raw_colors.shape)
			raw_colors = raw.postprocess(half_size=True, output_bps=16, user_flip=0, gamma = (1,1), no_auto_bright=True, use_camera_wb=False, use_auto_wb=False, user_wb = None, bright=1.0)

			# print(raw_colors.shape, raw_colors.dtype)

			return raw_colors
	else:
		import histogram_gap

		return histogram_gap.read_raw_correct_hist(path)#.astype('float32')


def display_image(i, z=1):
	disp_image = i.copy()

	disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

	low = np.percentile(disp_image, z)
	high = np.percentile(disp_image, 100 - z)
	disp_image = np.clip(disp_image, low, high)
	plt.imshow(disp_image)

def load_images(folder):	
	cache_filename = os.path.join(folder, 'images_stack.npy')

	if os.path.exists(cache_filename) and False:
		data = np.load(cache_filename)
	else:
		filenames = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(folder))))
		print('# filenames: ', len(filenames))

		data = None

		for i in tqdm.tqdm(range(len(filenames))):

			img = load_raw_image(filenames[i])
			# print(np.mean(img, axis=(0,1)))

			if i == 0:
				data = np.zeros((len(filenames),) + img.shape, dtype=img.dtype)
				print(data.shape, data.dtype)

			data[i] = img

		np.save(cache_filename, data)

	return data

def plot_std_dev(folder):
	data = load_images(folder)

	for channel in range(data.shape[3]):
		channel_stack = data[:, :, :, channel]
		std_dev_img = np.std(channel_stack, axis=0)
		mean_img = np.mean(channel_stack, axis=0)
		# print(std_dev_img)
		# print(np.mean(std_dev_img))
		if 1:
			plt.subplot(2, 2, 1)
			# plt.imshow(mean_img)
			display_image(mean_img, z=1)
			plt.title('mean')
			plt.subplot(2, 2, 2)
			display_image(std_dev_img, z=1)
			plt.title('std')

			plt.subplot(2, 2, 3)
			display_image(mean_img / std_dev_img, z=1)
			plt.title('mean / std')

			plt.subplot(2, 2, 4)
			bins = np.arange(np.min(channel_stack), np.max(channel_stack)+1)
			plt.hist(channel_stack.flatten(), bins = bins)
			plt.grid(True)
			plt.show()

		# skip = 10
		# for img_channel in channel_stack:
		# 	plt.scatter(img_channel.flatten()[::skip], mean_img.flatten()[::skip], alpha = 0.1, color='black', s=1)


		rnr = RadiusNeighborsRegressor(radius = 10, weights = 'distance')
		rnr.fit(np.expand_dims(mean_img.flatten(), axis=1), std_dev_img.flatten())

		line_x = np.arange(np.min(mean_img), np.max(mean_img)+1)
		line_y = rnr.predict(np.expand_dims(line_x, axis=1))


		fit = np.polyfit(mean_img.flatten(), std_dev_img.flatten(), deg=1)
		linear_y = np.polyval(fit, line_x)


		# for d in range(deg+1):
		# 	fits[y//n, :, channel, d] = section_fits[d]

		plt.scatter(mean_img.flatten(), std_dev_img.flatten(), alpha=0.1, color='black', s=1)
		plt.plot(line_x, line_y, 'r')
		plt.plot(line_x, linear_y, 'orange')
		plt.grid(True)
		plt.show()

def compare_averages(folder):
	data = load_images(folder)

	for channel in range(data.shape[3]):
		channel_stack = data[:, :, :, channel]


		img_mean = np.mean(channel_stack, axis=0)
		img_percentile_clip = scipy.stats.trim_mean(channel_stack, proportiontocut=0.1, axis=0)
		img_sigma_clip = np.mean(astropy.stats.sigma_clip(channel_stack, sigma=2, axis=0), axis=0)

		img_percentile_ratio = (img_mean / img_percentile_clip - 1) * 1E3
		img_sigma_ratio = (img_mean / img_sigma_clip - 1) * 1E3
		img_percent_sigma_ratio = (img_percentile_clip / img_sigma_clip - 1) * 1E3

		plt.subplot(3, 4, 1)
		display_image(img_mean)
		plt.title('mean')

		plt.subplot(3,4,2)
		display_image(img_percentile_clip)
		plt.title('percentile clip')

		plt.subplot(3,4,3)
		display_image(img_percentile_ratio)
		plt.title('percentile ratio')

		plt.subplot(3, 4, 4)
		plt.hist(img_percentile_ratio.flatten(), bins = 1000)


		plt.subplot(3, 4, 5)
		display_image(img_mean)
		plt.title('mean')

		plt.subplot(3,4,6)
		display_image(img_sigma_clip)
		plt.title('sigma clip')

		plt.subplot(3,4,7)
		display_image(img_sigma_ratio)
		plt.title('sigma ratio')

		plt.subplot(3, 4, 8)
		plt.hist(img_sigma_ratio.flatten(), bins = 1000)



		plt.subplot(3,4,9)
		display_image(img_percentile_clip)
		plt.title('percentile')

		plt.subplot(3,4,10)
		display_image(img_sigma_clip)
		plt.title('sigma clip')

		plt.subplot(3,4, 11)
		display_image(img_percent_sigma_ratio)
		plt.title('percent-sigma-ratio')

		plt.subplot(3, 4, 12)
		plt.hist(img_percent_sigma_ratio.flatten(), bins = 1000)

		plt.show()

def compare_error_vs_brightness(folder):
	data = load_images(folder)

	for channel in range(data.shape[3]):
		channel_stack = data[:, :, :, channel]

		img_mean = np.mean(channel_stack, axis=0)
		img_sigma_clip = np.mean(astropy.stats.sigma_clip(channel_stack, sigma=2, axis=0), axis=0)

		img_sigma_ratio = (img_mean / img_sigma_clip - 1) * 1E3

		x = np.arange(np.min(img_mean), np.max(img_mean)+1)
		bit_flip_change = 128 if channel == 1 else 256
		y_top = ((channel_stack.shape[0] * x) / (channel_stack.shape[0] * x - bit_flip_change)-1) * 1E3
		y_bottom = ((channel_stack.shape[0] * x) / (channel_stack.shape[0] * x + bit_flip_change)-1) * 1E3
		plt.plot(x, y_top, 'r')
		plt.plot(x, y_bottom, 'r')
		plt.scatter(img_mean.flatten(), img_sigma_ratio.flatten(), alpha=0.1, color='black', s=1)


		rnr = RadiusNeighborsRegressor(radius = 50, weights = 'distance')
		rnr.fit(np.expand_dims(img_mean.flatten(), axis=1), img_sigma_ratio.flatten())

		x = np.arange(np.min(img_mean), np.max(img_mean)+1)
		line_y = rnr.predict(np.expand_dims(x, axis=1))
		plt.plot(x, line_y, 'g')

		plt.grid(True)
		plt.show()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def pixel_vs_mean(folder):
	data = load_images(folder)

	for channel in range(data.shape[3]):
		channel_stack = data[:, :, :, channel]

		img_mean = np.mean(channel_stack, axis=0)

		skip = 10

		# for channel_img in channel_stack:
		# 	channel_sub_mean = channel_img - img_mean

		stack_sub_mean = channel_stack - img_mean

		#mean range 1500-1800
		if 1:
			mean_indices = np.unravel_index(np.where(np.abs(img_mean - 1650) < 150)[0], img_mean.shape)
			print(mean_indices[0].shape, mean_indices[1].shape)
			diffs = stack_sub_mean[:, mean_indices[0], mean_indices[1]].flatten()
		else:
			diffs = stack_sub_mean.flatten()

		bin_counts, bin_edges, patches = plt.hist(diffs[::skip], bins = 1000, density=True)
		bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
		mean, sigma = weighted_avg_and_std(bin_centers, bin_counts)

		plt.plot(bin_centers, 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((bin_centers - mean) / sigma)**2), 'r')

		plt.grid(True)
		plt.show()

def compare_multiple_stacks(folder):
	subfolders = os.listdir(folder)

	all_data = []

	for subfolder in tqdm.tqdm(subfolders):
		all_data.append(load_images(os.path.join(folder, subfolder)))

	all_data = np.array(all_data)
	print(all_data.shape)

	for channel in range(3):
		for subfolder_index in range(all_data.shape[0]):

			channel_stack = all_data[subfolder_index][:, :, :, channel]

			img_mean = np.mean(channel_stack, axis=0)
			img_sigma_clip = np.mean(astropy.stats.sigma_clip(channel_stack, sigma=2, axis=0), axis=0)

			img_sigma_ratio = (img_mean / img_sigma_clip - 1) * 1E3
			skip = 1
			flat_ratios = img_sigma_ratio.flatten()[::skip]
			mean_values = img_mean.flatten()[::skip]

			# plt.scatter(mean_values, flat_ratios, alpha=0.1, color='black', s=1)


			rnr = RadiusNeighborsRegressor(radius = 50, weights = 'uniform')
			rnr.fit(np.expand_dims(mean_values, axis=1), flat_ratios.flatten())

			x = np.arange(np.min(mean_values)+200, np.max(mean_values)+1-200, 10)
			line_y = rnr.predict(np.expand_dims(x, axis=1))
			plt.plot(x, line_y, label=str(subfolder_index))

		plt.legend()
		plt.grid(True)
		plt.show()


if __name__ == "__main__":

	#600mm
	# folder = 'F:/Pictures/Lightroom/2020/2020-02-29/flats1'

	#135mm
	# folder = 'F:/Pictures/Lightroom/2020/2020-03-15/untracked_sky_flats'
	# folder = 'F:/Pictures/Lightroom/2020/2020-03-01/135mm_computer_screen_flats/0.0333333'
	folder = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats/3'
	# folder = 'F:/Pictures/Lightroom/2020/2020-03-07/darks_135mm_30s_iso100'

	# plot_std_dev(folder)
	compare_averages(folder)
	# compare_error_vs_brightness(folder)
	# pixel_vs_mean(folder)




	folder = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats'

	# folder = 'F:/Pictures/Lightroom/2020/2020-02-29/flats'
	compare_multiple_stacks(folder)
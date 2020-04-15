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

from flats_progression import load_images

def load_raw_image(path):
	with rawpy.imread(path) as raw:

		# raw_colors = np.array(raw.raw_colors)
		# print(raw_colors.shape)
		raw_colors = raw.postprocess(half_size=True, output_bps=16, user_flip=0, gamma = (1,1), no_auto_bright=True, use_camera_wb=False, use_auto_wb=False, user_wb = None, bright=1.0)

		# print(raw_colors.shape, raw_colors.dtype)

		return raw_colors

def main():


	if 0:
		bias_image = load_raw_image('F:/2020/2020-02-26/bias_iso800/DSC05241.ARW').astype('float')
		bias_mean = np.mean(bias_image, axis=(0,1))
		print('bias mean: ', bias_mean)

	if 0:
		files = [('F:/2020/2020-02-26/linearity_test_iso800/DSC05320.ARW', 1/2000),
		('F:/2020/2020-02-26/linearity_test_iso800/DSC05321.ARW', 1/1000),
		('F:/2020/2020-02-26/linearity_test_iso800/DSC05322.ARW', 1/500),
		('F:/2020/2020-02-26/linearity_test_iso800/DSC05323.ARW', 1/250),
		('F:/2020/2020-02-26/linearity_test_iso800/DSC05324.ARW', 1/125),
		('F:/2020/2020-02-26/linearity_test_iso800/DSC05325.ARW', 1/60),
		# ('F:/2020/2020-02-26/linearity_test_iso800/DSC05326.ARW', 1/30),
		# ('F:/2020/2020-02-26/linearity_test_iso800/DSC05327.ARW', 1/15),
		# ('F:/2020/2020-02-26/linearity_test_iso800/DSC05328.ARW', 1/8),
		]

		filenames, exposure_times = zip(*files)
		print(filenames)
		images = []
		for filename in filenames:
			img = load_raw_image(filename).astype('float')

			img = gaussian_filter(img, mode='nearest', sigma=5)

			images.append(img)
		images = np.array(images)
		exposure_times = np.array(exposure_times)
	elif 1:
		# folders = ['F:/2020/2020-02-15/flats_1', 'F:/2020/2020-02-15/flats_2', 'F:/2020/2020-02-15/flats_3', 'F:/2020/2020-02-15/flats_4']; exposure_times = [1/60, 1/125, 1/250, 1/500]
		# folders = ['F:/2020/2020-02-20/flats_1', 'F:/2020/2020-02-20/flats_2', 'F:/2020/2020-02-20/flats_3', 'F:/2020/2020-02-20/flats_4', 'F:/2020/2020-02-20/flats_5'][::-1]; exposure_times = [1/3, 1/6, 1/13, 1/30, 1/60]
		# folders = ['F:/2020/2020-02-20/flats_2', 'F:/2020/2020-02-20/flats_3', 'F:/2020/2020-02-20/flats_4', 'F:/2020/2020-02-20/flats_5'][::-1]; exposure_times = [1/6, 1/13, 1/30, 1/60][::-1]
		
		# folders = ['F:/2020/2020-02-26/flats_1', 'F:/2020/2020-02-26/flats_2', 'F:/2020/2020-02-26/flats_3', 'F:/2020/2020-02-26/flats_4', 'F:/2020/2020-02-26/flats_5', 'F:/2020/2020-02-26/flats_6']; exposure_times = [1/2, 1/4, 1/8, 1/16, 1/30, 1/60]
		
		# folders = ['K:/Orion_600mm/flats1', 'K:/Orion_600mm/flats2', 'K:/Orion_600mm/flats3', 'K:/Orion_600mm/flats4', 'K:/Orion_600mm/flats5']; exposure_times = [1/100, 1/50, 1/25, 1/13, 1/8]

		# folders = ['F:/2020/2020-02-18/blue_sky_flats/1', 'F:/2020/2020-02-18/blue_sky_flats/2', 'F:/2020/2020-02-18/blue_sky_flats/3', 'F:/2020/2020-02-18/blue_sky_flats/4', 'F:/2020/2020-02-18/blue_sky_flats/5']
		
		base_folder = 'F:/2020/2020-03-01/135mm_computer_screen_flats'
		folders = []
		exposure_times = []
		for f in os.listdir(base_folder)[:-2]:
			folders.append(os.path.join(base_folder, f))
			exposure_times.append(float(f))

		images = []
		for folder in folders:
			print(folder)
			img = load_images(folder)
			img = gaussian_filter(img, mode='nearest', sigma=5)

			images.append(img)

		images = np.array(images)
		exposure_times = np.array(exposure_times)

		images = images[::3]
		exposure_times = exposure_times[::3]

	else:

		flat_filenames = ['K:/Orion_600mm/stacked_flats/flats1.tif',
			'K:/Orion_600mm/stacked_flats/flats2.tif',
			'K:/Orion_600mm/stacked_flats/flats3.tif',
			'K:/Orion_600mm/stacked_flats/flats4.tif',
			'K:/Orion_600mm/stacked_flats/flats5.tif',
			]

		exposure_times = [1/100, 1/50, 1/25, 1/13, 1/8]

		def gray_2_rgb(img):
			r = img[::2, ::2]
			g1 = img[1::2, ::2]
			g2 = img[::2, 1::2]
			b = img[1::2, 1::2]

			result = np.array([r, (g1 + g2)/2, b])
			result = np.transpose(result, (1, 2, 0))
			# result = np.concatenate([r, (g1+g2)/2, b], axis=2)
			print(result.shape)
			return result

		flat_images = [tiff.imread(fn) for fn in flat_filenames]
		images = np.array([gray_2_rgb(img) for img in flat_images])

		images = np.array([gaussian_filter(img, mode='nearest', sigma=5) for img in images])


	if 1:
		for channel in range(3):
			channel_pixels = images[:, :, :, channel]
			if 0:
				for img in channel_pixels:
					plt.hist(img.flatten(), bins = 100)
				plt.grid(True)
				plt.show()

			means = np.mean(channel_pixels, axis=(1,2))

			fit = np.polyfit(exposure_times, means, deg=1)

			fit_means = np.polyval(fit, exposure_times)
			plt.subplot(1, 2, 1)
			plt.plot(exposure_times, means)
			plt.plot(exposure_times, fit_means, 'r')
			plt.grid(True)
			plt.title(fit)

			plt.subplot(1, 2, 2)
			plt.semilogx(exposure_times, np.abs(means - fit_means))
			plt.grid(True)
			plt.title('errors')

			plt.show()

	xs = np.array([np.mean(i) for i in images])
	# xs = exposure_times


	all_residuals = 0
	deg = 1
	fits = np.zeros((images.shape[1], images.shape[2], images.shape[3], deg+1))
	for y in tqdm.tqdm(range(0, images.shape[1])):
		for channel in range(images.shape[3]):

			to_fit = images[:, y, :, channel]
			section_fits, residuals, rank, singular_values, rcond = np.polyfit(xs, to_fit, deg=deg, full=True)
			# print(residuals)
			all_residuals += np.sum(residuals)

			for d in range(deg+1):
				fits[y, :, channel, d] = section_fits[d]

	print('all residuals: ', all_residuals / np.prod(images.shape))
	print('y-intercepts: ', np.mean(fits[:, :, :, -1], axis=(0, 1)))

	for i in range(deg+1):
		for channel in range(3):
			plt.subplot(2, 2, channel+1)
			plt.imshow(fits[:, :, channel, i])
			plt.title(str(np.mean(fits[:, :, channel, i])))

		# plt.title('poly val: %d' % i)
		plt.show()


if __name__ == "__main__":
	main()
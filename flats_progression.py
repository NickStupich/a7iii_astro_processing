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

def load_raw_image(path):
	with rawpy.imread(path) as raw:

		# raw_colors = np.array(raw.raw_colors)
		# print(raw_colors.shape)
		raw_colors = raw.postprocess(half_size=True, output_bps=16, user_flip=0, gamma = (1,1), no_auto_bright=True, use_camera_wb=False, use_auto_wb=False, user_wb = None, bright=1.0)

		# print(raw_colors.shape, raw_colors.dtype)

		return raw_colors

def load_images(images_folder):

	use_cache = True

	cache_name = os.path.join(images_folder, 'mean_image.cache')
	if os.path.exists(cache_name) and use_cache:# and False:
		sum_image = pickle.load(open(cache_name, 'rb'))
	else:
		# sum_image = np.zeros((4024, 6048), dtype='float')
		sum_image = np.zeros((2012, 3012, 3), dtype='float')
		# sum_image = np.zeros((2010, 3011, 3), dtype='float')

		filenames = list(filter(lambda s: s.endswith('.ARW') or s.endswith('.CR2'), os.listdir(images_folder)))

		for img_fn in tqdm.tqdm(filenames):
				raw_colors = load_raw_image(os.path.join(images_folder, img_fn))
				sum_image += raw_colors

		sum_image = sum_image.astype('float64') / len(filenames)

		if use_cache:
			pickle.dump(sum_image, open(cache_name, 'wb'))

	return sum_image


def kappa_sigma_mean(c, kappa=2, num_iters = 5):
	c_next = c.copy()
	for i in range(num_iters):
		c_std = c_next.std()
		c_mean = c_next.mean()
		size = c_next.size
		critlower = c_mean - c_std*kappa
		critupper = c_mean + c_std*kappa
		c_next = c[(c > critlower) & (c < critupper)]
		delta = size-c_next.size
		if delta == 0:
			return c_mean

	return c_mean

def load_images_kappa_sigma(images_folder):

	use_cache = True

	cache_name = os.path.join(images_folder, 'kappa_sigma_mean_image.cache')
	if os.path.exists(cache_name) and use_cache:# and False:
		sum_image = pickle.load(open(cache_name, 'rb'))
	else:
		# sum_image = np.zeros((4024, 6048), dtype='float')
		# sum_image = np.zeros((2012, 3012, 3), dtype='float')
		# sum_image = np.zeros((2010, 3011, 3), dtype='float')

		filenames = list(filter(lambda s: s.endswith('.ARW') or s.endswith('.CR2'), os.listdir(images_folder)))

		all_images = np.zeros((len(filenames), 2012, 3012, 3), dtype='float')

		for i, img_fn in enumerate(tqdm.tqdm(filenames)):
			raw_colors = load_raw_image(os.path.join(images_folder, img_fn))
			# sum_image += raw_colors
			all_images[i] = raw_colors

		# sum_image = sum_image.astype('float64') / len(filenames)

		sum_image = np.zeros((2012, 3012, 3), dtype='float')
		for y in tqdm.tqdm(range(sum_image.shape[0])):
			for x in range(sum_image.shape[1]):
				for c in range(3):
					sum_image[y,x,c] = kappa_sigma_mean(all_images[:, y, x, c])

		if use_cache:
			pickle.dump(sum_image, open(cache_name, 'wb'))

	return sum_image

def clipped_display(img, z=5):

	disp_image = img.copy()

	disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

	for i in range(3):
		low = np.percentile(disp_image[:, :, i], z)
		high = np.percentile(disp_image[:, :, i], 100 - z)


		disp_image[:, :, i] = np.clip(disp_image[:, :, i], low, high)
		disp_image[:, :, i] -= np.min(disp_image[:, :, i])
		disp_image[:, :, i] /= np.max(disp_image[:, :, i])

	plt.imshow(disp_image)
	plt.show()

def process_folders(folders):

	sigma = False

	flat_images = []
	for folder in folders:
		print(folder)

		if sigma:
			mean_image = load_images_kappa_sigma(folder)
		else:
			mean_image = load_images(folder)

		print(np.mean(mean_image, axis = (0, 1)))
		# plt.imshow(mean_image / (2**16))
		# plt.show()

		# mean_image /= np.mean(mean_image)

		flat_images.append(mean_image)

	flat_images = np.array(flat_images)
	color_means = np.mean(flat_images, axis=(1, 2))
	print('color means: ', color_means)

	if 0:
		plt.plot(color_means[:, 0], 'r')
		plt.plot(color_means[:, 1], 'g')
		plt.plot(color_means[:, 2], 'b')
		plt.grid(True)
		plt.show()

	# diffs = flat_images[-1] / flat_images[0]
	# diffs /= (np.mean(diffs) * 2)

	# clipped_display(diffs, z=5)

	# plt.imshow(diffs)
	# plt.show()

	for i in range(flat_images.shape[0]):
		flat_images[i] = gaussian_filter(flat_images[i], mode='nearest', sigma=5)


	deg = 1

	if sigma:
		fits_name = os.path.join(folders[0], 'fits%d_sigma.cache' % deg)
	else:
		fits_name = os.path.join(folders[0], 'fits%d.cache' % deg)

	if os.path.exists(fits_name) and False:
		fits = pickle.load(open(fits_name, 'rb'))
	else:
		n = 1

		fits = np.zeros((int(np.ceil(flat_images.shape[1] / n)), int(np.ceil(flat_images.shape[2] / n)), flat_images.shape[3], deg+1))

		# xs = np.arange(0, flat_images.shape[0])
		xs = np.array([np.mean(i) for i in flat_images])
		channel_means = np.mean(flat_images, axis=(1,2))

		# print(xs, xs.shape)
		# print(channel_means.shape)
		for y in tqdm.tqdm(range(0, flat_images.shape[1], n)):
			# for x in range(0, flat_images.shape[2], n):
			# 	for channel in range(flat_images.shape[3]):
			# 		vals = flat_images[:, y, x, channel]
			# 		# print(vals, vals.shape)
			# 		# plt.plot(vals / vals[0], alpha = 0.01, c = ['r', 'g', 'b'][channel])

			# 		fit = np.polyfit(xs, vals, deg=1)
			# 		# fits.append(fit)
			# 		fits[y//n][x//n][channel] = fit

			# to_fit = np.reshape(flat_images[:, y, :, :], (flat_images.shape[0], flat_images.shape[2] * flat_images.shape[3]))
			# section_fits = np.polyfit(xs, to_fit, deg=deg)
			# # print(section_fits.shape)
			# section_fits = np.reshape(section_fits, (deg+1, flat_images.shape[2], flat_images.shape[3]))
			# # print(section_fits.shape)
			# for d in range(deg+1):
			# 	fits[y//n, :, :, d] = section_fits[d, :, :]


			for channel in range(flat_images.shape[3]):
				# xs = np.array([np.mean(i[:, :, channel]) for i in flat_images])
				xs = channel_means[:, channel]
				# print()
				to_fit = flat_images[:, y, :, channel]
				section_fits = np.polyfit(xs, to_fit, deg=deg)	

				for d in range(deg+1):
					fits[y//n, :, channel, d] = section_fits[d]


			# section_fits = np.reshape(section_fits, ())


		fits = np.array(fits)

		pickle.dump(fits, open(fits_name, 'wb'))

	# print(fits.shape)

	yint = fits[:, :, :, 1]
	slopes = fits[:, :, :, 0]


	# plt.hist(yint[::10, ::10, 0])
	# plt.hist(yint[::10, ::10, 1])
	# plt.hist(yint[::10, ::10, 2])

	# plt.hist(slopes[:, :, 0])
	# plt.hist(slopes[:, :, 1])
	# plt.hist(slopes[:, :, 2])



	# plt.imshow(yint)

	# plt.grid(True)
	# plt.show()

	if 0:
		clipped_display(yint, z=5)
		clipped_display(slopes)

	if 0:
		plt.subplot(2, 2, 1)
		plt.imshow(yint[:, :, 0])
		plt.subplot(2, 2, 2)
		plt.imshow(yint[:, :, 1])
		plt.subplot(2, 2, 3)
		plt.imshow(yint[:, :, 2])
		plt.show()

		plt.imshow(np.mean(slopes, axis=2))
		plt.show()


	if 0:
		for i in range(3):
			x = np.linspace(0, 1, slopes.shape[1])
			y = np.linspace(0, 1, slopes.shape[0])
			X, Y = np.meshgrid(x, y, copy=False)
			
			X = X.flatten()
			Y = Y.flatten()

			A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
			B = slopes[:, :, i].flatten()

			coeff, r, rank, s = np.linalg.lstsq(A, B)
			# coeff[0] = 0

			fit = np.reshape(A.dot(coeff), slopes.shape[:2])
			print(coeff)
			print(A.shape, coeff.shape, fit.shape)

			# plt.imshow(fit)
			# plt.show()
			slopes[:, :, i] /= fit

	for i in range(fits.shape[3]):
		print('i: %d  mean: %e' % (i, np.mean(fits[:, :, :, i])))

	# print('mean y-int: ', np.mean(yint))
	# print('mean slope: ', np.mean(slopes))

	# test_path = 'F:/2020/2020-02-15/flats_2/DSC03200.ARW'
	# test_path = 'F:/2020/2020-02-15/flats_2/DSC03200.ARW'
	# test_path = 'F:/2020/2020-02-19/flame_horsehead_600mm/DSC03918.ARW'
	# test_path = 'F:/2020/2020-02-20/pleiades/DSC04397.ARW'
	test_path = 'F:/2020/2020-02-18/orion_135mm/DSC03645.ARW'
	test_image = load_raw_image(test_path).astype('float')
	for _ in [1]:



	# for test_image in flat_images:

		# corrected_image = (test_image - yint) / slopes
		if deg == 1:
			corrected_image = (test_image - fits[:, :, :, 1]) / fits[:, :, :, 0]

		elif deg == 2:
			a = fits[:, :, :, 2]
			b = fits[:, :, :, 1]
			c = fits[:, :, :, 0]

			corrected_image = -b + (np.sqrt(b*b - 4*a*c*test_image)) / 2*a

		if 1:
			plt.subplot(2, 1, 1)
			plt.hist(test_image[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
			plt.hist(test_image[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
			plt.hist(test_image[:, :, 2].flatten(), bins = 100, color='b', histtype='step')
			plt.yscale('log', nonposy='clip')

			plt.subplot(2, 1, 2)
			plt.hist(corrected_image[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
			plt.hist(corrected_image[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
			plt.hist(corrected_image[:, :, 2].flatten(), bins = 100, color='b', histtype='step')
			plt.yscale('log', nonposy='clip')

			plt.show()


			plt.subplot(2, 1, 1)
			plt.imshow(test_image / 16000)
			plt.subplot(2, 1, 2)
			plt.imshow(corrected_image / 16000)
			plt.show()

			z = 5
			clipped_display(test_image, z=z)
			clipped_display(corrected_image, z=z)

			tiff.imwrite('corrected.tif', corrected_image.astype('uint16'))

	run_folder = 'F:/2020/2020-02-18/orion_135mm'
	out_folder = 'F:/2020/2020-02-18/orion_135mm_processed'
	filenames = list(filter(lambda s: s.endswith('.ARW'), os.listdir(run_folder)))
	for filename in tqdm.tqdm(filenames):
		test_path = os.path.join(run_folder, filename)
		test_image = load_raw_image(test_path).astype('float')
		corrected_image = (test_image - fits[:, :, :, 1]) / fits[:, :, :, 0]

		out_fn = os.path.join(out_folder, filename)
		tiff.imwrite(out_fn, corrected_image.astype('uint16'))

def linear_interpolate_flats(flat_folders):

	flats = np.array([load_images(folder) for folder in folders])

	ratio = flats[0] / flats[-2]

	# plt.imshow(ratio / np.max(ratio))
	# plt.show()

	clipped_display(ratio)

	flat_means = np.array([np.mean(flat, axis=(0, 1)) for flat in flats])


	print('flat means: ', flat_means)

	run_folder = 'F:/2020/2020-02-18/orion_135mm'
	out_folder = 'F:/2020/2020-02-18/orion_135mm_processed2'
	filenames = list(filter(lambda s: s.endswith('.ARW'), os.listdir(run_folder)))
	for filename in tqdm.tqdm(filenames):
		test_path = os.path.join(run_folder, filename)
		test_image = load_raw_image(test_path).astype('float')

		#todo: separately by color channel or all together?
		#by part of the picture? sort of like above?

		test_image_means = np.mean(test_image, axis=(0, 1))
		# test_image_means = np.median(test_image, axis=(0, 1))
		# print('test image means: ', test_image_means)

		# contributions = (bright_flat_means - test_image_means) / (bright_flat_means - dark_flat_means)
		# print('contributions: ', contributions)

		interpolated_flat = np.zeros_like(test_image)

		for channel in range(3):
			channel_mean = test_image_means[channel]
			flat_channel_means = flat_means[:, channel]
			indices = np.arange(0, len(flat_channel_means))
			interp = np.interp(channel_mean, flat_channel_means, indices)
			# print('channel mean: ', channel_mean)
			# print('flat channel means: ', flat_channel_means)
			# print('interpolation: ', interp)

			low_index = int(np.floor(interp))
			high_index = int(np.ceil(interp))

			low_weight = 1 - (interp - low_index)
			high_weight = 1 - (high_index - interp)

			# print(channel, low_index, high_index, low_weight, high_weight, type(channel), type(low_index))

			interpolated_flat[:, :, channel] = low_weight * flats[low_index, :, :, channel] + high_weight * flats[high_index, :, :, channel]

		interpolated_flat /= np.mean(interpolated_flat)
		# print('mean flat: ', np.mean(interpolated_flat))
		# clipped_display(interpolated_flat)


		corrected_image = test_image / interpolated_flat
		# clipped_display(test_image)
		# clipped_display(corrected_image)


		# corrected_image = (test_image - fits[:, :, :, 1]) / fits[:, :, :, 0]

		out_fn = os.path.join(out_folder, filename)
		tiff.imwrite(out_fn, np.clip((corrected_image * 10), 0, 65535).astype('uint16'))

		# break

def compare_flats_to_stack():
	lights_folder = 'F:/2020/2020-02-20/pleiades'
	# mean_lights_image = load_images(lights_folder)
	# clipped_display(mean_lights_image, z=5)

	cache_name = os.path.join(lights_folder, 'removed_stars_mean_image.cache')
	if os.path.exists(cache_name):
		sum_image = pickle.load(open(cache_name, 'rb'))
	else:
		sum_image = None
		filenames = list(filter(lambda s: s.endswith('.ARW'), os.listdir(lights_folder)))[:10]
		for filename in tqdm.tqdm(filenames):
			test_path = os.path.join(lights_folder, filename)
			test_image = load_raw_image(test_path).astype('float')

			if sum_image is None:
				sum_image = np.zeros_like(test_image)

			img_8bit = np.mean(test_image, axis=2)
			img_8bit = (img_8bit / (np.max(img_8bit) / 255)).astype('uint8')

			bw_image = np.mean(test_image, axis=2)

			thresh_niblack = threshold_niblack(bw_image, window_size=51, k=0.0)
			stars_mask = bw_image > (thresh_niblack + 250)

			for _ in range(5):
				stars_mask = dilation(stars_mask)

			mask_copy = stars_mask.copy()
			working_image = test_image.copy()

			while np.sum(mask_copy) > 0:
				mask_copy = erosion(mask_copy)
				for channel in range(3):
					working_image[:, :, channel] = erosion(working_image[:, :, channel])

			if 0:
				plt.imshow(np.log(np.mean(test_image, axis=2)))
				plt.show()

				plt.imshow(np.log(np.mean(working_image, axis=2)))
				plt.show()

			filled_image = np.zeros_like(test_image)
			for channel in range(3):
				filled_image[:, :, channel] = test_image[:, :, channel] * (1 - stars_mask) + working_image[:, :, channel] * stars_mask

			sum_image += filled_image

			if 0:
				plt.imshow(np.log(np.mean(filled_image, axis=2)))
				plt.show()

			if 0:
				plt.subplot(3, 1, 1)
				# plt.imshow(img_8bit)
				# stars_thresh = cv2.adaptiveThreshold(img_8bit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 1)
				# plt.imshow(test_image / np.max(test_image))
				plt.imshow(bw_image / np.max(bw_image))

				# stars_thresh = threshold_adaptive(test_image, 201, offset = 10)

				# thresh_sauvola = threshold_sauvola(bw_image, window_size=21)
				# stars_thresh = bw_image > thresh_sauvola
				# print(thresh_sauvola)

				plt.subplot(3, 1, 2)
				plt.imshow(stars_mask  / np.max(stars_mask))

				plt.subplot(3, 1, 3)
				plt.imshow(filled_image / np.max(filled_image))

				plt.show()


		sum_image /= len(filenames)
		pickle.dump(sum_image, open(cache_name, 'wb'))

	plt.imshow(sum_image[:, :, 0])
	plt.show()

		# stars_filled = cv2.inpaint(bw_image, stars_thresh, 3, cv2.INPAINT_TELEA)


	# plt.hist(mean_lights_image.flatten(), bins = 1000)
	# plt.show()

if __name__ == "__main__":
	# folders = ['F:/2020/2020-02-15/flats_1', 'F:/2020/2020-02-15/flats_2', 'F:/2020/2020-02-15/flats_3', 'F:/2020/2020-02-15/flats_4']
	folders = ['F:/2020/2020-02-20/flats_1', 'F:/2020/2020-02-20/flats_2', 'F:/2020/2020-02-20/flats_3', 'F:/2020/2020-02-20/flats_4', 'F:/2020/2020-02-20/flats_5'][::-1]

	# folders = ['F:/2020/2020-02-18/blue_sky_flats/1', 'F:/2020/2020-02-18/blue_sky_flats/2', 'F:/2020/2020-02-18/blue_sky_flats/3', 'F:/2020/2020-02-18/blue_sky_flats/4', 'F:/2020/2020-02-18/blue_sky_flats/5']

	# process_folders(folders)

	# linear_interpolate_flats(folders)
	compare_flats_to_stack()
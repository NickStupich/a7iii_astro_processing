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
from optimize_flats_weighting import *

def twoD_Gaussian(locs, amplitude, sigma, xo, yo, offset, sx, sy, amplitude2, sigma2):
    x, y = locs                                                 
    xo = float(xo)                                                              
    yo = float(yo)            
    g = offset + sx * x + sy * y + amplitude * np.exp(-(sigma**2) * ((x-xo)**2 + (y-yo)**2)) + amplitude2 * np.exp(-(sigma2**2) * ((x-xo)**2 + (y-yo)**2))
    return g.ravel()

def get_optical_center(img):

    # size_pixels = 5

    # position_int = (int(position[0]), int(position[1]))

    # x = np.linspace(position_int[0] - size_pixels, position_int[0] + size_pixels, size_pixels*2 )
    # y = np.linspace(position_int[1] - size_pixels, position_int[1] + size_pixels, size_pixels*2 )

    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])


    x, y = np.meshgrid(x, y)

    # sub_img = img[position_int[1] - size_pixels: position_int[1] + size_pixels, position_int[0] - size_pixels: position_int[0] + size_pixels]

    xy = np.vstack((x, y))

    initial_guess = (np.max(img)/5, 1E-3, #amplitude, sigma
    	img.shape[1]/2, img.shape[0]/2, #x0, y0
    	0, #offset
    	0, 0, #sx, sy
    	np.max(img)/10, 1E-3, #amplitude2, sigma2
    	)

    guess_plot = twoD_Gaussian((x, y), *initial_guess)

    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, (x, y), img.ravel(), p0=initial_guess)
    
    data_fitted = twoD_Gaussian((x, y), *popt).astype('float32')
    print(initial_guess, popt)

    center = (popt[2], popt[3])
    print(center)

    if 1: 
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.contour(data_fitted.reshape(x.shape), 8)
        plt.title('img and fit')
        plt.subplot(2, 2, 2)
        plt.imshow(data_fitted.reshape(img.shape))
        plt.title('fit')

        plt.subplot(2, 2, 3)
        diff = img - data_fitted.reshape(img.shape)
        z = 1
        low = np.percentile(diff, z)
        high = np.percentile(diff, 100 - z)
        diff = np.clip(diff, low, high)

        plt.imshow(diff)
        plt.title('difference: %.2e' % np.std(diff))

        plt.show() 

    return center

def get_radial_pixel_means(test_channel, rescale=True, center = None, plot=False):

	if center is None:
		cy = test_channel.shape[0] // 2 #+ 155
		cx = test_channel.shape[1] // 2 #+ 100
		# cy = 1225
		# cx = 1616
	else:
		cx, cy = center
		print('center: ', center)

	max_radius = int(np.ceil(np.sqrt(max(cx, test_channel.shape[1] - cx)**2 + max(cy, test_channel.shape[0] - cy)**2)))
	# print('max radius: ', max_radius)

	pixels_by_radius = [[] for _ in range(max_radius)]

	for y in tqdm.tqdm(range(test_channel.shape[0])):
		for x in range(test_channel.shape[1]):
			r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
			pixels_by_radius[r].append(test_channel[y,x])

			if plot:
				if r == 500 or r == 50 or r == 1000:
					test_channel[y,x] *= 5

	def radius_mean_func(l):
		l2 = scipy.stats.trimboth(l, proportiontocut = 0.2)
		return np.mean(l2), scipy.stats.mstats.sem(l2)
		# return np.mean(l), scipy.stats.mstats.sem(l)

	# radius_mean_func = lambda x: scipy.stats.trim_mean(x, proportiontocut = 0.2)
	# radius_mean_func = np.mean

	pixel_means_by_radius, pixel_stds_by_radius = map(np.array, zip(*[radius_mean_func(a) for a in pixels_by_radius]))

	scale = np.std(pixel_means_by_radius)
	if rescale:
		pixel_means_by_radius = (pixel_means_by_radius - np.mean(pixel_means_by_radius)) / scale
		pixel_stds_by_radius /= scale


	if plot:
		plt.subplot(3, 1, 1)
		display_image(test_channel)

		plt.subplot(3, 1, 2)
		plt.plot(pixel_means_by_radius)
		plt.plot(pixel_means_by_radius - 2*pixel_stds_by_radius)
		plt.plot(pixel_means_by_radius + 2*pixel_stds_by_radius)

		plt.subplot(3, 1, 3)
		plt.hist(pixels_by_radius[1000], bins = 20)
		plt.title('radius = 1000')

		plt.show()

	return pixel_means_by_radius, pixel_stds_by_radius

def apply_radial_correction(test_channel, radial_mult, center = None):
	if center is None:
		cy = test_channel.shape[0] // 2 #+ 155
		cx = test_channel.shape[1] // 2 #+ 100
		# cy = 1225
		# cx = 1616
	else:
		cx, cy = center
		print('center: ', center)

	output_channel = np.zeros_like(test_channel)

	max_radius = int(np.ceil(np.sqrt(max(cx, test_channel.shape[1] - cx)**2 + max(cy, test_channel.shape[0] - cy)**2)))
	# print('max radius: ', max_radius)

	pixels_by_radius = [[] for _ in range(max_radius)]

	for y in tqdm.tqdm(range(test_channel.shape[0])):
		for x in range(test_channel.shape[1]):
			r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
			output_channel[y,x] = test_channel[y,x] * radial_mult[r]

	return output_channel

def radial_brightness_comparison():

	if 0:
		flat_filenames = [
			'K:/orion_135mm_syntheticflats/computer_flats/0.05.tif',
			'K:/orion_135mm_syntheticflats/computer_flats/0.1.tif',
			'K:/orion_135mm_syntheticflats/computer_flats/0.2.tif',
			'K:/orion_135mm_syntheticflats/computer_flats/0.4.tif',
			'K:/orion_135mm_syntheticflats/computer_flats/0.8.tif',
			'K:/orion_135mm_syntheticflats/computer_flats/1.6.tif',
		]
	elif 0:
		flat_filenames = [
		'K:/Orion_135mm/flats/flats1_gray.tif',
		'K:/Orion_135mm/flats/flats2_gray.tif',
		'K:/Orion_135mm/flats/flats3_gray.tif',
		'K:/Orion_135mm/flats/flats4_gray.tif',
		'K:/Orion_135mm/flats/flats5_gray.tif',
	]
	elif 1:	
		flat_filenames = [
		'K:/orion_135mm_syntheticflats/blue_sky_flats/1.tif',
		'K:/orion_135mm_syntheticflats/blue_sky_flats/2.tif',
		'K:/orion_135mm_syntheticflats/blue_sky_flats/3.tif',
		'K:/orion_135mm_syntheticflats/blue_sky_flats/4.tif',
		'K:/orion_135mm_syntheticflats/blue_sky_flats/5.tif',
	]
	else:	
		flat_filenames = [
		'K:/orion_135mm_syntheticflats/03-15-orion-unreg-flat.tif',
		'K:/orion_135mm_syntheticflats/03-15-untracked-sky-flat.tif',
	]


	# dark_filename = 'K:/orion_135mm_syntheticflats/computer_flats/darks_stack_30s_iso100.tif',

	# test_stacked_image = 'K:/orion_135mm_syntheticflats/computer_flats/unaligned_mean_darkcal_light.tif'
	test_stacked_image = 'K:/orion_135mm_syntheticflats/computer_flats/unaligned_uncal_light.tif'

	flat_images = [load_gray_tiff(fn) for fn in flat_filenames]
	flat_images_rgb = np.array([extract_channel_image(img) for img in flat_images])

	# dark_img = load_gray_tiff(dark_filename)

	test_img = load_gray_tiff(test_stacked_image)		
	# print(np.mean(dark_img), np.mean(test_img))
	# test_img -= dark_img
	test_img = np.clip(test_img, 0, np.inf)
	test_img_rgb = extract_channel_image(test_img)

	optimized_flat_rgb = np.zeros_like(test_img_rgb)
	for channel in range(4):

		channel_flats = flat_images_rgb[:, channel]
		test_channel = test_img_rgb[channel]

		flats_mean_by_radius, flats_std_by_radius = zip(*[get_radial_pixel_means(channel_flat) for channel_flat in channel_flats])
		test_mean_by_radius, test_std_by_radius = get_radial_pixel_means(test_channel)

		mean_all_by_radius = np.mean(np.concatenate((flats_mean_by_radius, [test_mean_by_radius])), axis=0)
		# mean_all_by_radius *= 0
		sigma = 2


		plt.plot(test_mean_by_radius - mean_all_by_radius, c = 'b', label='test image')
		# plt.plot(test_mean_by_radius - mean_all_by_radius - sigma*test_std_by_radius, '--', c = 'b', label='test image')
		# plt.plot(test_mean_by_radius - mean_all_by_radius + sigma*test_std_by_radius, '--', c = 'b', label='test image')

		cs = ['g', 'r', 'c', 'y', 'm']
		for i, (f_mean, f_std)  in enumerate(zip(flats_mean_by_radius, flats_std_by_radius)):
			plt.plot(f_mean - mean_all_by_radius, c = cs[i], label='flat %d' % i)
			# plt.plot(f_mean - mean_all_by_radius - sigma*f_std, '--', c = cs[i], label='flat %d' % i)
			# plt.plot(f_mean - mean_all_by_radius + sigma*f_std, '--', c = cs[i], label='flat %d' % i)


		plt.grid(True)
		plt.legend()
		plt.show()

def display_image(img):
	disp_image = remove_gradient(img)
	disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)
	# disp_image = img.copy()

	# z = 1
	# low = np.percentile(disp_image, z)
	# high = np.percentile(disp_image, 100 - z)
	z = 0.01
	low = 1 - z
	high = 1 + z
	disp_image = np.clip(disp_image, low, high)
	plt.imshow(disp_image)
	plt.title(str(np.std(img)))

def radial_brightness_flat_matching():

	target_intensity = 'K:/orion_135mm_syntheticflats/computer_flats/unaligned_uncal_light.tif'

	flat_image = 'K:/orion_135mm_syntheticflats/blue_sky_flats/3.tif'
	# flat_image = 'K:/orion_135mm_syntheticflats/blue_sky_flats/2.tif'
	# flat_image = 'K:/orion_135mm_syntheticflats/blue_sky_flats/4.tif'

	target_img = load_gray_tiff(target_intensity)		
	target_img_rgb = extract_channel_image(target_img)

	flat_img = load_gray_tiff(flat_image)		
	flat_img_rgb = extract_channel_image(flat_img)

	output_flat = np.zeros_like(flat_img_rgb)

	for channel in range(4):	
		target_channel = target_img_rgb[channel]

		raw_flat_channel = flat_img_rgb[channel]


		target_channel = remove_gradient(target_channel)
		flat_channel = remove_gradient(raw_flat_channel)

		if 0:
			target_center = get_optical_center(target_channel)
			flat_center = get_optical_center(flat_channel)

			avg_center = ((target_center[0] + flat_center[0])/2, (target_center[1] + flat_center[1])/2)
		else:
			avg_center = (1615, 1179)
			# avg_center = (1567, 1059)

		if 0:
			target_mean_by_radius, target_std_by_radius = get_radial_pixel_means(target_channel, rescale=False, center = avg_center)
			flat_mean_by_radius, flat_std_by_radius = get_radial_pixel_means(flat_channel, rescale=False, center = avg_center)

			#TODO: filter. possibly with the std too?
			flat_radial_mult_factor = target_mean_by_radius / flat_mean_by_radius
		else:
			image_ratio = target_channel / flat_channel
			ratio_mean_by_radius, ratio_std_by_radius = get_radial_pixel_means(image_ratio, rescale=False, center = avg_center)

			flat_radial_mult_factor = ratio_mean_by_radius


		# plt.subplot(2, 1, 1)
		# plt.plot(target_mean_by_radius)
		# plt.plot(flat_mean_by_radius)
		# plt.grid(True)

		# plt.subplot(2, 1, 2)
		# plt.plot(flat_radial_mult_factor)
		# plt.show()


		corrected_channel = apply_radial_correction(flat_channel, flat_radial_mult_factor, center = avg_center)
		raw_corrected_channel = apply_radial_correction(raw_flat_channel, flat_radial_mult_factor, center = avg_center)

		calibrated_test_channel = target_channel / corrected_channel
		
		avg_center = (1567, 1059)
		corrected_mean_by_radius, corrected_std_by_radius = get_radial_pixel_means(calibrated_test_channel, rescale=False, center = avg_center, plot=False)
		

		double_calibrated_flat = apply_radial_correction(corrected_channel, corrected_mean_by_radius, center = avg_center)
		raw_corrected_channel = apply_radial_correction(raw_corrected_channel, corrected_mean_by_radius, center = avg_center)
		
		double_calibrated_test_channel = target_channel / double_calibrated_flat

		double_corrected_mean_by_radius, double_corrected_std_by_radius = get_radial_pixel_means(double_calibrated_test_channel, rescale=False, center = avg_center)

		if 0:
			plt.subplot(2, 2, 1)
			# plt.imshow(corrected_channel)
			display_image(target_channel/flat_channel)

			plt.subplot(2, 2, 2)
			display_image(calibrated_test_channel)

			plt.subplot(2, 2, 3)

			plt.plot(corrected_mean_by_radius, label='after 1st correction')
			plt.grid(True)


			plt.subplot(2, 2, 4)
			display_image(double_calibrated_test_channel)

			plt.subplot(2, 2, 3)
			plt.plot(double_corrected_mean_by_radius, label='double corrected')
			plt.legend()

			plt.show()


		output_flat[channel] = raw_corrected_channel


	flat_optimized_flat = flatten_channel_image(output_flat)

	tiff.imwrite('radial_optimized_flat_for_stack.tif', flat_optimized_flat.astype('float32'))

if __name__ == "__main__":
	# radial_brightness_comparison()
	radial_brightness_flat_matching()
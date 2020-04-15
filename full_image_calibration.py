import os
import tifffile as tiff
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import astropy

import histogram_gap
from tiff_conversion import load_dark, IMG_MAX
from interpolate_flats import load_gray_tiff, extract_channel_image, flatten_channel_image
from interpolate_fixed_flats import get_exposure_matched_flat, display_image, remove_gradient

import full_flat_sequence

def load_raw_image(filename, master_dark):
	img = histogram_gap.read_raw_correct_hist(filename)

	if master_dark is not None:
		img -= master_dark
	else:
		img -= 512
		print('no bias/dark frame')

	img = np.clip(img, 0, np.inf)
	return img

def get_mean_flat(folder, bias_or_dark_frame, force_reload = False):

	cache_filename = os.path.join(folder, 'master_flat.tif')

	if os.path.exists(cache_filename) and not force_reload:
		print('loading flat from cache...', cache_filename)
		result = tiff.imread(cache_filename)
		return result

	else:

		filenames = list(map(lambda s2: os.path.join(folder, s2), filter(lambda s: s.endswith('.ARW'), os.listdir(folder))))
		print('filenames in folder: ', folder)
		print(filenames)

		image_stack = None
		for i, filename in enumerate(filenames):
			raw_img = load_raw_image(filename, bias_or_dark_frame)

			if image_stack is None:
				image_stack = np.zeros((len(filenames), ) + raw_img.shape, dtype=raw_img.dtype)
				print(image_stack.shape)

			image_stack[i] = raw_img

		#TODO: better than straight mean
		result = np.mean(image_stack, axis=0)
		# result = np.mean(astropy.stats.sigma_clip(image_stack, sigma=2, axis=0), axis=0)


		tiff.imwrite(cache_filename, result)

	return result

def load_flats_from_subfolders(folder, bias_or_dark_frame):
	subfolders = np.array(list(filter(lambda s: os.path.isdir(s), map(lambda s2: os.path.join(folder, s2), os.listdir(folder)))))

	if len(subfolders) == 0:
		print('no subfolders, assuming just 1 set of flats in passed directory')
		subfolders = np.array([folder])

	try:
		order = np.argsort([int(fn.split(os.path.sep)[-1]) for fn in subfolders])
		subfolders = subfolders[order]
		print(subfolders)
	except :
		print('failed to presort folders')

	result = None

	for i, subfolder in enumerate(subfolders):
		flat_frame = get_mean_flat(subfolder, bias_or_dark_frame)

		noise_levels = [calc_relative_image_noise_level(flat_frame[:, :, channel]) for channel in range(flat_frame.shape[-1])]
		print('noise: ', noise_levels)

		if result is None:
			result = np.zeros((len(subfolders),) + flat_frame.shape, dtype=flat_frame.dtype)

		result[i] = flat_frame

	flat_brightnesses = [np.mean(frame) for frame in result]
	order = np.argsort(flat_brightnesses)
	print(flat_brightnesses, order)

	result = result[order]
	subfolders = subfolders[order]

	result /= IMG_MAX

	return result, subfolders

def calc_relative_image_noise_level(img):
	img_mean = np.mean(img)
	dx = np.mean(np.abs(np.diff(img, axis=0)))
	dy = np.mean(np.abs(np.diff(img, axis=1)))

	result = (dx + dy) / img_mean

	return result

def calc_flats_noise(flats):
	for flat in flats:
		# for channel in range(flat.shape[-1]):
		channel = 0
		channel = flat[:, :, channel]
		noise = calc_relative_image_noise_level(channel)
		print(noise)

		# exit(0)

def main():
	"""
	workflow:
calc bias frame	 - sigma mean
	remove histogram gaps
	folder input
	cache

calc dark frame - sigma mean
	remove histogram gaps
	folder input
	cache

calc series of flat frames - sigma mean
	remove histogram gaps
	subtract bias
	folder input
	cache

for each light frame:
	remove histogram gaps
	subtract dark frame
	for each channel:
		pick optimal weighting of flat frames
		divide by flat

	save image as tif

 

-> pixinsight
	debayer
	register
	integrate
	"""
	if 0:
		lights_in_folder = 'K:/orion_135mm_bothnights/lights_in'

		darks_folder = 'K:/orion_135mm_bothnights/darks'
		bias_folder = 'K:/orion_135mm_bothnights/bias'

		calibrated_lights_out_folder = 'K:/orion_135mm_bothnights/lights_out'
		# flats_folder = 'F:/2020/2020-04-06/blue_sky_flats'
		flats_folder = 'F:/Pictures/Lightroom/2020/2020-02-18/blue_sky_flats'


		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'

	elif 0:
		lights_in_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/flame_horsehead_600mm'

		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/bias'
		
		flats_folder = 'F:/Pictures/Lightroom/2020/2020-02-20/flats'
		# flats_folder = 'F:/2020/2020-04-07/flats_600mm'
		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'

		calibrated_lights_out_folder = 'K:/flame_horsehead_600mm/lights_out_flat_sequence'
	elif 0:
		lights_in_folder = 'F:/Pictures/Lightroom/2020/2020-02-29/orion_600mm'

		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/bias'
		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		
		flats_folder = 'F:/Pictures/Lightroom/2020/2020-02-29/flats'
		# flats_folder = 'F:/2020/2020-04-07/flats_600mm'

		calibrated_lights_out_folder = 'K:/orion_600mm/lights_out'
	elif 0:
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/bias'
		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		# flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'

		flats_folder = 'F:/2020/2020-04-11/flats_135mm_tshirtwall'

		# lights_in_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/1'
		# calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_1'

		# lights_in_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/2'
		# calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_2'

		lights_in_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/3'
		calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_3'

	else:
		darks_folder = 'F:/Pictures/Lightroom/2020/2020-03-07/darks_135mm_30s_iso100'
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-03-07/bias_135mm_iso100'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
		flats_folder = 'F:/2020/2020-04-12/flats_135mm_clothsky'

		lights_in_folder = 'F:/2020/2020-04-12/coma_cluster_135mm'
		calibrated_lights_out_folder = 'K:/coma_cluster_135mm/lights_out'


	#Todo: make smarter mean. also cache
	master_dark = load_dark(darks_folder)
	master_bias = load_dark(bias_folder)

	flat_images, flat_subfolders = load_flats_from_subfolders(flats_folder, master_bias)

	flat_images_rgb = np.transpose(flat_images, (0, 3, 1, 2))
	print(flat_images.shape, flat_images_rgb.shape)


	filenames = list(filter(lambda s: s.endswith('.ARW'), os.listdir(lights_in_folder)))
	print('number of lights: ', len(filenames))

	for filename in tqdm.tqdm(filenames):
		full_path = os.path.join(lights_in_folder, filename)
		output_path = os.path.join(calibrated_lights_out_folder, filename.rstrip('.ARW') + '.tif')

		# if os.path.exists(output_path): continue

		img_minus_dark = load_raw_image(full_path, master_dark)


		img_minus_dark_rgb = np.transpose(img_minus_dark, (2, 0, 1)) / IMG_MAX

		# for channel in range(4):
		# 	plt.imshow(img_minus_dark_rgb[channel, 1990//2:2010//2, 2990//2:3010//2])
		# 	plt.show()

		print(img_minus_dark.shape, img_minus_dark_rgb.shape)

		# print(img_minus_dark_rgb.shape)
		if 0:
			calibrated_flat_img_rgb, exposure_index = get_exposure_matched_flat(flat_images_rgb, img_minus_dark_rgb)
		else:
			calibrated_flat_img_rgb = full_flat_sequence.get_flat_and_bandingfix_flat(flat_images_rgb, img_minus_dark_rgb, flats_progression_folder, master_bias)


		if calibrated_flat_img_rgb is None: 
			print("***Failed to find matching flat")
			continue
		# print('exposure index: ', exposure_index)

		calibrated_img_rgb = img_minus_dark_rgb / calibrated_flat_img_rgb
		# calibrated_img_rgb = img_minus_dark_rgb

		print('calibrated rgb: ', calibrated_img_rgb.shape)
		calibrated_img = flatten_channel_image(calibrated_img_rgb)

		tiff.imwrite(output_path, calibrated_img.astype('float32'))

		# exit(0)



if __name__ == "__main__":
	main()


	"""todo: 
		tighter spaced flat frames
			automatic stacking & cal of flat frames

		caching throughout

		power lines removal/python stacking
		horizontal banding reduction. pre-registering. fix in flats??? visible in fft @ freq=670/1000 (0.5 * vertical resolution)  python vs pixinsight?
		


	"""
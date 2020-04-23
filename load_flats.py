import os
import numpy as np
import tqdm
import tifffile as tiff
import matplotlib.pyplot as plt
import astropy

import histogram_gap

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
			raw_img = histogram_gap.load_raw_image(filename, bias_or_dark_frame)

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

		# noise_levels = [calc_relative_image_noise_level(flat_frame[:, :, channel]) for channel in range(flat_frame.shape[-1])]
		# print('noise: ', noise_levels)

		if result is None:
			result = np.zeros((len(subfolders),) + flat_frame.shape, dtype=flat_frame.dtype)

		result[i] = flat_frame

	flat_brightnesses = [np.mean(frame) for frame in result]
	order = np.argsort(flat_brightnesses)
	print(flat_brightnesses, order)

	result = result[order]
	subfolders = subfolders[order]

	return result, subfolders
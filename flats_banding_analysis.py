import os
import tifffile as tiff
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import astropy
from scipy.ndimage import gaussian_filter

import histogram_gap
from tiff_conversion import load_dark, IMG_MAX
from interpolate_flats import load_gray_tiff, extract_channel_image, flatten_channel_image
from interpolate_fixed_flats import get_exposure_matched_flat, display_image, remove_gradient
from full_image_calibration import load_flats_from_subfolders



def plot_flats_banding_progression(folder, bias_or_dark_frame):
	flats, folder_names = load_flats_from_subfolders(folder, bias_or_dark_frame)
	# calc_flats_noise(flats)
	# exit(0)
	flat_means = np.mean(flats, axis=(1, 2, 3))
	for i in range(len(flat_means)):
		flats[i] /= flat_means[i]

	channel = 0
	overall_mean_flat = np.mean(flats, axis=0)
	plt.imshow(overall_mean_flat[:, :, channel])

	for i in range(len(flat_means)):
		cal_flat = flats[i] / overall_mean_flat

		# plt.imshow(cal_flat[:, :, channel])
		z = 5
		disp_image = cal_flat[:, :, channel].copy()
		disp_image = remove_gradient(disp_image, quadratic=False)

		disp_image = gaussian_filter(disp_image, mode='nearest', sigma=5)

		low = np.percentile(disp_image, z)
		high = np.percentile(disp_image, 100 - z)
		disp_image = np.clip(disp_image, low, high)
		plt.imshow(disp_image)
		plt.title(folder_names[i])
		plt.show()

def main():

	"""
	workflow idea:
		clean sensor real good
		take really fine grained 'mapping' of weird bands once. shitton of images, long processing

		each night:
			take 1 set of flats close to light frames
				determine mean flat -> fine grained flats map
					- get an image of dust spots, focus differences, etc

				for each image:	
					get fine grained flat map frame
						multiply fine grained flat * nightly flat map = real flat
						calibrate light frame using flat



	"""

	bias_folder = 'K:/orion_135mm_bothnights/bias'
	
	if 1:
		flats_folder = 'F:/2020/2020-04-06/blue_sky_flats'
	elif 0:
		flats_folder = 'F:/2020/2020-04-07/flats_600mm'
	else:
		flats_folder = 'F:/Pictures/Lightroom/2020/2020-03-01/135mm_computer_screen_flats'

	master_bias = load_dark(bias_folder)

	flat_images = load_flats_from_subfolders(flats_folder, master_bias)
	if 1:
		plot_flats_banding_progression(flats_folder, master_bias)
		exit(0)

if __name__ == "__main__":
	main()
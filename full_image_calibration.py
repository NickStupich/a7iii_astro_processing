import os
import tifffile as tiff
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import astropy

import histogram_gap
from tiff_conversion import load_dark
from interpolate_flats import load_gray_tiff, extract_channel_image, flatten_channel_image
from interpolate_fixed_flats import get_exposure_matched_flat, display_image, remove_gradient
from load_flats import load_flats_from_subfolders
import full_flat_sequence
import subimg_full_flat_sequence
import pixinsight_preprocess

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
	if 1:
		lights_in_folder = 'K:/orion_135mm_bothnights/lights_in'

		darks_folder = 'K:/orion_135mm_bothnights/darks'
		bias_folder = 'K:/orion_135mm_bothnights/bias'

		output_folder = 'K:/orion_135mm_bothnights2'
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
		bias_folder = 'F:/2020/2020-04-17/bias_iso800'
		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		
		flats_folder = 'F:/Pictures/Lightroom/2020/2020-02-29/flats'
		# flats_folder = 'F:/2020/2020-04-07/flats_600mm'

		calibrated_lights_out_folder = 'K:/orion_600mm/lights_out'
	elif 0:
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/bias'
		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		# flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
		
		# flats_folder = 'F:/2020/2020-04-06/blue_sky_flats'
		flats_folder = 'F:/2020/2020-04-11/flats_135mm_tshirtwall'

		# lights_in_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/1'
		# calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_1'

		# lights_in_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/2'
		# calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_2'

		lights_in_folder = 'F:/2020/2020-04-10/casseiopeia_pano_135mm/1'
		# calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_3'
		calibrated_lights_out_folder = 'K:/casseiopeia_pano/lights_1_blurredflats'
	elif 0:
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/bias'
		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'

		flats_folder = 'F:/2020/2020-04-16/rho_ophiuchi/flats'
		lights_in_folder = 'F:/2020/2020-04-16/rho_ophiuchi/lights'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'		

		calibrated_lights_out_folder = 'K:/rho_oph/lights'
	elif 0:
		bias_folder = 'F:/2020/2020-04-17/bias_iso800'
		darks_folder = 'F:/2020/2020-04-17/darks_iso800_2mins'

		flats_folder = 'F:/2020/2020-04-17/flats_135mmf4_v2'
		lights_in_folder = 'F:/2020/2020-04-17/coma_cluster_135mmf4'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'		

		calibrated_lights_out_folder = 'K:/coma_135mm_f4/lights'
	elif 0:

		lights_in_folder = 'F:/2020/2020-04-15/leo_triplet_600mm'

		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		bias_folder = 'F:/2020/2020-04-17/bias_iso800'
		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		
		flats_folder = 'F:/2020/2020-04-15/flats'

		calibrated_lights_out_folder = 'K:/leo_triplet_600mm/lights_out'
	elif 0:

		lights_in_folder = 'F:/Pictures/Lightroom/2020_2018/2018-02-01/andromeda_600mm'

		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		bias_folder = 'F:/2020/2020-04-17/bias_iso800'
		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		
		flats_folder = 'F:/Pictures/Lightroom/2020_2018/2018-02-01/flats_600mm'

		calibrated_lights_out_folder = 'K:/andromeda_600mm/lights_out'
	elif 0:
		darks_folder = 'F:/Pictures/Lightroom/2020/2020-03-07/darks_135mm_30s_iso100'
		bias_folder = 'F:/Pictures/Lightroom/2020/2020-03-07/bias_135mm_iso100'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'
		flats_folder = 'F:/2020/2020-04-12/flats_135mm_clothsky'

		lights_in_folder = 'F:/2020/2020-04-12/coma_cluster_135mm'
		calibrated_lights_out_folder = 'K:/coma_cluster_135mm/lights_out'
	elif 0:
		lights_in_folder = 'F:/Pictures/Lightroom/2020/2020-02-20/pleiades'

		darks_folder = 'F:/Pictures/Lightroom/2020/2020-02-19/darks'
		bias_folder = 'F:/2020/2020-04-17/bias_iso800'
		flats_progression_folder = 'F:/2020/2020-04-09/shirt_dusk_flats_progression'
		
		flats_folder = 'F:/Pictures/Lightroom/2020/2020-02-20/pleiades_flats'

		calibrated_lights_out_folder = 'Z:/astro_processing/pleiades/lights_pleiades_flats'
	elif 1:
		darks_folder = 'K:/orion_135mm_bothnights/darks'
		bias_folder = 'K:/orion_135mm_bothnights/bias'
		flats_progression_folder = 'F:/2020/2020-04-11/135mm_sky_flats_progression'		
		flats_folder = 'F:/2020/2020-04-22/24mm_flats'

		# lights_in_folder = 'F:/2020/2020-04-11/rice_lake_nightscape/tracked'
		# calibrated_lights_out_folder = 'K:/rice_lake/tracked_lights'

		lights_in_folder = 'F:/2020/2020-04-11/rice_lake_nightscape/untracked'
		calibrated_lights_out_folder = 'K:/rice_lake/untracked_lights'


	if not os.path.exists(output_folder): os.mkdir(output_folder)
	calibrated_lights_out_folder = os.path.join(output_folder, 'calibrated')
	if not os.path.exists(calibrated_lights_out_folder): os.mkdir(calibrated_lights_out_folder)

	#Todo: make smarter mean. also cache
	master_dark = load_dark(darks_folder)
	master_bias = load_dark(bias_folder)

	flat_images, flat_subfolders = load_flats_from_subfolders(flats_folder, master_bias)

	flat_images_rgb = np.transpose(flat_images, (0, 3, 1, 2))
	print(flat_images.shape, flat_images_rgb.shape)


	filenames = list(filter(lambda s: s.endswith('.ARW'), os.listdir(lights_in_folder)))[:7]
	print('number of lights: ', len(filenames))

	for filename in tqdm.tqdm(filenames):
		full_path = os.path.join(lights_in_folder, filename)
		output_path = os.path.join(calibrated_lights_out_folder, filename.rstrip('.ARW') + '.tif')
		# if os.path.exists(output_path): continue

		img_minus_dark = histogram_gap.load_raw_image(full_path, master_dark)
		img_minus_dark_rgb = np.transpose(img_minus_dark, (2, 0, 1))

		if 0:
			calibrated_flat_img_rgb, exposure_index = get_exposure_matched_flat(flat_images_rgb, img_minus_dark_rgb)
		elif 0:
			calibrated_flat_img_rgb = full_flat_sequence.get_flat_and_bandingfix_flat(flat_images_rgb, img_minus_dark_rgb, flats_progression_folder, master_bias)
		elif 0:
			calibrated_flat_img_rgb = subimg_full_flat_sequence.get_subimg_matched_flat2(flat_images_rgb, img_minus_dark_rgb, flats_progression_folder, master_bias)
		elif 1:
			calibrated_flat_img_rgb = subimg_full_flat_sequence.get_flat_and_subimg_matched_flat(flat_images_rgb, img_minus_dark_rgb, flats_progression_folder, master_bias)
		else:
			calibrated_flat_img_rgb = 1

		if calibrated_flat_img_rgb is None: 
			print("***Failed to find matching flat")
			continue

		calibrated_img_rgb = img_minus_dark_rgb / calibrated_flat_img_rgb
		calibrated_img_rgb = np.clip(calibrated_img_rgb, 0, 1)
		# calibrated_img_rgb = img_minus_dark_rgb

		# print('calibrated rgb: ', calibrated_img_rgb.shape)
		calibrated_img = flatten_channel_image(calibrated_img_rgb)

		tiff.imwrite(output_path, calibrated_img.astype('float32'))

		# exit(0)

	pi_script_path = pixinsight_preprocess.create_pixinsight_preprocess_script(output_folder, calibrated_lights_out_folder, filenames)
	cmd = '"C:/Program Files/PixInsight/bin/PixInsight.exe" --run=%s' % pi_script_path

	os.system(cmd)

if __name__ == "__main__":
	main()
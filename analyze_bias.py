import rawpy
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from scipy.ndimage import gaussian_filter
import pickle


def load_images(images_folder):

	cache_name = os.path.join(images_folder, 'mean_image.cache')
	if os.path.exists(cache_name):
		sum_image = pickle.load(open(cache_name, 'rb'))
	else:
		# sum_image = np.zeros((4024, 6048), dtype='float')
		sum_image = np.zeros((2012, 3012, 3), dtype='float')
		# sum_image = np.zeros((2010, 3011, 3), dtype='float')

		filenames = list(filter(lambda s: s.endswith('.ARW') or s.endswith('.CR2'), os.listdir(images_folder)))

		for img_fn in tqdm.tqdm(filenames):
			with rawpy.imread(os.path.join(images_folder, img_fn)) as raw:

				# raw_colors = np.array(raw.raw_image)
				# raw_colors = np.array(raw.raw_colors)
				raw_colors = raw.postprocess(half_size=True, output_bps=16, user_flip=0)
				print(raw_colors.shape, raw_colors.dtype)

				# r = raw_colors[::2, ::2]
				# g1 = raw_colors[1::2, ::2]
				# g2 = raw_colors[::2, 1::2]
				# g = (g1 + g2) / 2
				# b = raw_colors[1::2, 1::2]

				# img = np.concatenate()

				sum_image += raw_colors

		sum_image = sum_image.astype('float64') / len(filenames)
		pickle.dump(sum_image, open(cache_name, 'wb'))

	return sum_image

def show_bands(sum_image):
	sum_image = gaussian_filter(sum_image, mode='nearest', sigma=5)
	sum_image = sum_image[6:-6, 6:-6]

	if 0:
		plt.hist(sum_image[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
		plt.hist(sum_image[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
		plt.hist(sum_image[:, :, 2].flatten(), bins = 100, color='b', histtype='step')

		plt.yscale('log', nonposy='clip')
		plt.show()

	if 1:
		for i in range(3):
			x = np.linspace(0, 1, sum_image.shape[1])
			y = np.linspace(0, 1, sum_image.shape[0])
			X, Y = np.meshgrid(x, y, copy=False)
			
			X = X.flatten()
			Y = Y.flatten()

			A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
			B = sum_image[:, :, i].flatten()

			coeff, r, rank, s = np.linalg.lstsq(A, B)
			# coeff[0] = 0

			fit = np.reshape(A.dot(coeff), sum_image.shape[:2])
			print(coeff)
			print(A.shape, coeff.shape, fit.shape)

			# plt.imshow(fit)
			# plt.show()
			sum_image[:, :, i] /= fit

	if 1:
		plt.hist(sum_image[:, :, 0].flatten(), bins = 100, color='r', histtype='step')
		plt.hist(sum_image[:, :, 1].flatten(), bins = 100, color='g', histtype='step')
		plt.hist(sum_image[:, :, 2].flatten(), bins = 100, color='b', histtype='step')

		plt.yscale('log', nonposy='clip')
		plt.show()

	if 1:
		intensity = np.mean(sum_image, axis=2)

		for i in range(3):
			sum_image[:, :, i] /= intensity


	if 1:
		disp_image = sum_image.copy()
		for i in range(3):
			z = 5
			low = np.percentile(disp_image[:, :, i], z)
			high = np.percentile(disp_image[:, :, i], 100 - z)


			disp_image[:, :, i] = np.clip(disp_image[:, :, i], low, high)
			disp_image[:, :, i] -= np.min(disp_image[:, :, i])
			disp_image[:, :, i] /= np.max(disp_image[:, :, i])

		plt.imshow(disp_image)
		plt.show()


	left_half = sum_image[:, :sum_image.shape[1] // 2]
	right_half = sum_image[:, sum_image.shape[1] // 2:]

	if 0:
		for half, name in [(left_half, 'left'), (right_half, 'right')]:

			for i in range(3):
				row_means = np.mean(half[:, :, i], axis=1)

				plt.plot(row_means, color = ['r', 'g', 'b'][i], label=name)

		plt.grid(True)
		plt.legend()
		plt.show()

	for i in range(3):
		diff_img = left_half[:, :, i] - right_half[:, :, i]
		diff = np.mean(diff_img, axis=1)
		dev = np.std(diff_img, axis=1)
		plt.plot(diff, color = ['r', 'g', 'b'][i])
		# plt.plot(diff - dev, linestyle='dashed', color = ['r', 'g', 'b'][i])
		# plt.plot(diff + dev, linestyle='dashed', color = ['r', 'g', 'b'][i])

	plt.grid(True)
	plt.show()

def main():
	# images_folder = 'images/bias'
	images_folder = 'images/darks_iso800_30s'
	# images_folder = 'images/lights_nolens'
	# images_folder = 'images/flats_600mm'
	# images_folder = 'images/flats_600mm_2'

	# images_folder = 'F:/2018/2018-02-01/andromeda_600mm'

	# images_folder = 'F:/2020/2020-02-15/flats_1'
	# images_folder = 'F:/2020/2020-02-15/flats_2'
	# images_folder = 'F:/2020/2020-02-15/flats_3'
	# images_folder = 'F:/2020/2020-02-15/flats_4'

	# images_folder = 'images/canon_50mm_flats'
	# images_folder = 'images/feb18_1'


	# images_folder = 'F:/2020/2020-02-18/blue_sky_flats/4'
	# images_folder = 'F:/2020/2020-02-18/darks'
	# images_folder = 'F:/2020/2020-02-18/orion_135mm/'
	# images_folder = 'F:/2020/2020-02-19/600mm_skytrail_flats'
	# images_folder = 'F:/2020/2020-02-19/flame_horsehead_600mm'
	# images_folder = 'F:/2020/2020-02-20/flats_5'
	# images_folder = 'F:/2020/2020-02-20/pleiades'

	sum_image = load_images(images_folder)
	show_bands(sum_image)
	
if __name__ == "__main__":
	main()
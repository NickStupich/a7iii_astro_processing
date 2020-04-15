import rawpy
import matplotlib.pyplot as plt
import numpy as np

# overexposed_path = 'images/single shot mech/DSC01978.ARW'
# black_path = 'images/single shot mech/DSC01979.ARW'

# overexposed_path = 'images/single shot elec/DSC01984.ARW'
# black_path = 'images/single shot elec/DSC01985.ARW'


# overexposed_path = 'images/continuous hi elec/DSC01987.ARW'
# black_path = 'images/continuous hi elec/DSC01988.ARW'


overexposed_path = 'images/continuous hi+ elec/DSC02034.ARW'
black_path = 'images/continuous hi+ elec/DSC02035.ARW'

with rawpy.imread(overexposed_path) as raw:

	raw_colors = np.array(raw.raw_image)
	print(raw_colors.shape, raw_colors.dtype)
	print(np.mean(raw_colors))
	print(np.min(raw_colors), np.max(raw_colors))
	bits = np.log(np.max(raw_colors)) / np.log(2)
	print('bits: ', bits)

	plt.hist(raw_colors.flatten(), bins = 100)
	plt.yscale('log', nonposy='clip')
	plt.title('overexposed image')
	plt.show()




with rawpy.imread(black_path) as raw:
	raw_colors = np.array(raw.raw_image)
	read_noise = np.std(raw_colors)
	print('noise std dev: ', read_noise)

	plt.imshow(np.clip(raw_colors, 480, 540))
	plt.grid(True)
	plt.show()

	plt.hist(raw_colors.flatten(), bins = np.arange(470, 580))
	plt.yscale('log', nonposy='clip')
	plt.title('black image')
	plt.show()


print('edr: ', bits - np.log(read_noise) / np.log(2))
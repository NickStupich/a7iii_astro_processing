import rawpy
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from scipy.ndimage import gaussian_filter
import pickle
import tifffile as tiff
from skimage.transform import warp_polar, rotate



fn = 'images/orion_stacked_135mm.TIF'
# img_pil = Image.open(fn)
# img = np.array(img_pil)

raw_img = tiff.imread(fn)

scale = np.max(raw_img)
img = raw_img.astype('float64') / scale

print(img.shape, img.dtype)

radius = np.max(img.shape)//2
radius = np.sqrt(img.shape[0]**2 + img.shape[1]**2)/2 - 200
print(radius)
polar_img = warp_polar(img, radius = radius, multichannel=True)

plt.subplot(4, 1, 1)
plt.imshow(img)

plt.subplot(4, 1, 2)
plt.imshow(polar_img)


plt.subplot(4, 1, 3)
channel_means = np.zeros((polar_img.shape[1], 3))
for channel in range(3):
	for r in range(polar_img.shape[1]):
		# x = np.mean(polar_img[:, r, channel])
		vals = polar_img[:, r, channel]
		filtered_vals = vals[np.where(vals > 0)]
		
		# x = np.mean(filtered_vals)
		x = np.median(filtered_vals)

		channel_means[r, channel] = x
	plt.plot(channel_means[:, channel], color = ['r', 'g', 'b'][channel])


subtract_image = np.zeros_like(img)

target_gray = np.mean(img)
print('target: ', target_gray)
cx = subtract_image.shape[1] // 2
cy = subtract_image.shape[0] // 2

for y in tqdm.tqdm(range(subtract_image.shape[0])):
	for x in range(subtract_image.shape[1]):
		r = np.sqrt((cy - y)**2 + (cx - x)**2)
		if r > channel_means.shape[0]:
			r = channel_means.shape[0]-1
		# print((channel_means[int(r)] - target_gray))
		subtract_image[y, x] = img[y, x] - (channel_means[int(r)] - target_gray)

output_image = (subtract_image * scale).astype(raw_img.dtype)
tiff.imsave('output.tiff', output_image)

plt.subplot(4, 1, 4)
plt.imshow(subtract_image)

plt.show()

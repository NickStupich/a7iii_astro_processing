import rawpy
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from scipy.ndimage import gaussian_filter
import pickle

from analyze_bias import load_images, show_bands

flats_folders = ['F:/2020/2020-02-15/flats_1',
			'F:/2020/2020-02-15/flats_2',
			'F:/2020/2020-02-15/flats_3',
			'F:/2020/2020-02-15/flats_4']

for folder in flats_folders:

	sum_image = load_images(images_folder)

	
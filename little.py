import glob
import numpy as np
from pathlib import Path
import os
import readwrite as rw
from PIL import Image

folder = "/media/tom/SMART-1TO/Dani/max_proj_HELM"

combined_folder = os.path.join(folder, "combined", "")
if not os.path.exists(combined_folder):
    os.mkdir(combined_folder)

file_list = []
for path in Path(folder).glob('*.tif'):
    file_list.append(path)

file_list = sorted(file_list)


for index in range(len(file_list)//3):
    red_file = file_list[3*index]
    green_file = file_list[3*index+1]
    blue_file = file_list[3*index+2]

    
    final = np.zeros((1024,1024,3), dtype=np.uint8)

    red_tiff = rw.open_tiff(red_file)
    red_tiff = red_tiff.astype(np.float)
    red_tiff = red_tiff - np.min(red_tiff)
    red_tiff = red_tiff*255/np.max(red_tiff)
    red_tiff = red_tiff.astype(np.uint8)
    final[:,:,0] = red_tiff[:,:,0]

    green_tiff = rw.open_tiff(green_file)
    green_tiff = green_tiff.astype(np.float)
    green_tiff = green_tiff - np.min(green_tiff)
    green_tiff = green_tiff*255/np.max(green_tiff)
    green_tiff = green_tiff.astype(np.uint8)
    final[:,:,1] = green_tiff[:,:,0]

    blue_tiff = rw.open_tiff(blue_file)
    blue_tiff = blue_tiff.astype(np.float)
    blue_tiff = blue_tiff - np.min(blue_tiff)
    blue_tiff = blue_tiff*255/np.max(blue_tiff)
    blue_tiff = blue_tiff.astype(np.uint8)
    final[:,:,2] = blue_tiff[:,:,0]


    split_name = str(red_file.resolve()).split('/')
    suffix = split_name[-1][:-8]+'_HELM.png'
    combined_file_name = combined_folder + suffix
    im = Image.fromarray(final)
    im.save(combined_file_name)


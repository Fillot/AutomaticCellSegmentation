import os
from pathlib import Path
import glob

import numpy as np
from PIL import Image
from skimage.external import tifffile as tif
from cellpose import models, io, utils



def prepare_folders(root_folder):
    """Simply create all the folders we'll be using, if they don't already exists
    Should work on windows but not sure."""
    split_folder_name = os.path.join(root_folder, "split_channels", "")
    max_folder_name = os.path.join(root_folder, "max_projections", "")
    cp_outlines_folder_name = os.path.join(root_folder, "outlines_cellpose", "")
    FQ_outlines_folder_name = os.path.join(root_folder, "outlines_FQ", "")
    masks_folder_name = os.path.join(root_folder, "masks", "")
    if not os.path.exists(split_folder_name):
        os.mkdir(split_folder_name)
    if not os.path.exists(max_folder_name):
        os.mkdir(max_folder_name)
    if not os.path.exists(cp_outlines_folder_name):
        os.mkdir(cp_outlines_folder_name)
    if not os.path.exists(FQ_outlines_folder_name):
        os.mkdir(FQ_outlines_folder_name)
    if not os.path.exists(masks_folder_name):
        os.mkdir(masks_folder_name)

def open_tiff(file_path):
# from https://stackoverflow.com/questions/37722139/load-a-tiff-stack-in-a-numpy-array-with-python
    dataset = Image.open(file_path)
    h,w = np.shape(dataset)
    tiffarray = np.zeros((h,w,dataset.n_frames), dtype='uint16')
    for i in range(dataset.n_frames):
        dataset.seek(i)
        tiffarray[:,:,i] = np.array(dataset)
    return tiffarray

def open_png(file_path):
    dataset = Image.open(file_path)
    img_array = np.array(dataset)
    return img_array


def open_three_channel_tiff(file_path):
    dataset = Image.open(file_path)
    h,w = np.shape(dataset)
    first_channel = np.zeros((h,w,dataset.n_frames//3), dtype='uint16')
    second_channel = np.zeros((h,w,dataset.n_frames//3), dtype='uint16')
    third_channel = np.zeros((h,w,dataset.n_frames//3), dtype='uint16')
    for i in range(dataset.n_frames//3):
        dataset.seek(3*i)
        first_channel[:,:,i] = np.array(dataset)
        dataset.seek(3*i+1)
        second_channel[:,:,i] = np.array(dataset)
        dataset.seek(3*i+2)
        third_channel[:,:,i] = np.array(dataset)
    return first_channel, second_channel, third_channel

def maxproj(stack):
    if len(np.shape(stack))!=3:
        raise Exception(f"Input is dimension {len(np.shape(stack))} but must be exactly of dimension 3 for the max projection to work.")
    max_proj = np.max(stack, axis=2)
    return max_proj

def save_stack_as_tiff(stack, name):
    tif.imsave(name, stack, byteorder='>', imagej=True, metadata={'mode': 'composite'})

def save_single_channel_as_tiff(stack, name):
    """Saving a stack as grayscale image in imagej
    ATT: most of the docs says tif.imsave doesn't handle 16bit, yet this works."""
    tif.imsave(name, stack, byteorder='>', imagej=True, metadata={'mode': 'grayscale'})

def save_mask_as_tiff(mask, name):
    byteorder='>'
    mask = np.asarray(mask, byteorder+'f', 'C')
    save_stack_as_tiff(mask, name)

def save_outlines_as_text(mask, name):
    outlines = utils.outlines_list(mask)
    io.outlines_to_text(name, outlines)

def save_splitted_channels(path, split_folder_name, red, green, blue):
    splited = path.name.split("_")
    out_name_split = splited[0]+"_"+splited[1]
    # reordering because the stack is YXZ but imagej wants ZYX
    red = np.moveaxis(red, -1, 0)
    red.shape = (1,red.shape[0], 1, red.shape[1],red.shape[2] , 1)
    green = np.moveaxis(green, -1, 0)
    green.shape = (1,green.shape[0], 1, green.shape[1],green.shape[2] , 1)
    blue = np.moveaxis(blue, -1, 0)
    blue.shape = (1,blue.shape[0], 1, blue.shape[1],blue.shape[2] , 1)

    # little switcharoo of the channels so that the order of the channel is always rgb
    channel_list = [red, green, blue]
    for i in range(1,4):
        suffix = f"_CH{i}.tif"
        save_single_channel_as_tiff(
            channel_list[i-1], os.path.join(split_folder_name, out_name_split+suffix))

    return red, green, blue

def max_tiff_as_png(root_folder, search_key='*_MAX.tif'):

    abs_path_list = []
    path_names = []

    png_proj_folder_name = os.path.join(root_folder, "max_proj_png")

    if not os.path.exists(png_proj_folder_name):
        os.mkdir(png_proj_folder_name)

    for path in Path(root_folder).rglob(search_key):
        abs_path_list.append(path)
        path_names.append(path.name)

    for i, path in enumerate(abs_path_list):
        tiff_img = open_tiff(path)
        
        for j in range(3):
            channel = tiff_img[:,:,j].astype(np.float)
            channel = channel - np.min(channel)
            channel = channel*255/np.max(channel)
            channel = channel.astype(np.uint8)
            tiff_img[:,:,j] = channel

        tiff_img = tiff_img.astype(np.uint8)
        im = Image.fromarray(tiff_img)
        file_suffix = path_names[i][:-4]+".png"
        file_name = os.path.join(png_proj_folder_name, file_suffix)
        im.save(file_name)



def tiff_split(path, split_folder_name, channels=[1,2,3]):
    """
    Supposedly opens a tif, splits, save and returns the different channel
    in the order RGB. Breaks down as soon as the image isn't rgb, so it's garbage. 
    """
    splited = path.name.split("_")
    out_name_split = splited[0]+"_"+splited[1]

    # splitting the channels
    tiff_CH1, tiff_CH2, tiff_CH3 = open_three_channel_tiff(path)
    tiff_list = [tiff_CH1, tiff_CH2, tiff_CH3]
    for i, c in enumerate(channels):
        if i==0:
            red = tiff_list[c-1]
        if i==1:
            green = tiff_list[c-1]
        if i ==2:
            blue = tiff_list[c-1]
    return red, green, blue

def split_and_project(
    root_folder, all_tiff = True, red=3, green=1, blue=2,
    summary=True):
    """
    Loads each tiff, saves the splitted channels and max projections of each.
    """
    split_folder_name = os.path.join(root_folder, "split_channels", "")
    max_folder_name = os.path.join(root_folder, "max_projections", "")
    summary_file_name = os.path.join(root_folder, "summary.txt")

    # normally the folders already exists
    if not os.path.exists(split_folder_name):
        os.mkdir(split_folder_name)
    if not os.path.exists(max_folder_name):
        os.mkdir(max_folder_name)

    path_list = []
    if all_tiff:
        search_key='*.tif'
    else:
        search_key=all_tiff
    
    for path in Path(root_folder).rglob(search_key):
        path_list.append(path)

    for path in path_list:
        print(f"Processing {path.name}")

        splited = path.name.split("_")
        out_name_max = splited[0]+"_"+splited[1]+"_MAX.tif"
        out_name_split = splited[0]+"_"+splited[1]

        tiff_red, tiff_green, tiff_blue = tiff_split(
            path, split_folder_name, channels=[red, green, blue])

        save_splitted_channels(
            path, split_folder_name, tiff_red, tiff_green, tiff_blue)
        max_projections = \
            [maxproj(tiff_red),maxproj(tiff_green),maxproj(tiff_blue)]

        shape = np.shape(tiff_blue)
        combined = np.zeros((3,shape[0],shape[1]), dtype='uint16')
        for i in range(3):
            combined[i,:,:] = max_projections[i]
            
        save_stack_as_tiff(combined, os.path.join(max_folder_name, out_name_max))

        if summary:
            # log the histogram of values for each channel and image
            # log which image have been processed already? 
            pass
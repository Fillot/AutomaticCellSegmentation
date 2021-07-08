import os
from pathlib import Path
import glob

import numpy as np
from PIL import Image
from skimage.external import tifffile as tif
from cellpose import models, io, plot, utils
import readwrite as rw

def logging(root_folder):
    # little log to verify which images have been treated already 
    # and skip them
    summary_file_name = os.path.join(root_folder, "summary.txt")
    max_folder_name = os.path.join(root_folder, "max_proj_HELM", "")
    untreated_images = []

    if os.path.isfile(summary_file_name):
        print("Logging file exists:")
        with open(summary_file_name, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                split = line.split(',')
                if int(split[1])==0:
                    untreated_images.append(split[0])
        print(f"Found {len(untreated_images)} images to segment.")
    else:
        print("Logging file doesn't exist")
        for path in Path(max_folder_name).rglob('*_HELM.png'):
            untreated_images.append(os.path.join(max_folder_name, path.name))
        untreated_images = sorted(untreated_images)
        with open(summary_file_name, 'w+') as log_file:
            for image_path in untreated_images:
                log_file.write(image_path+",0\n")
    return untreated_images


def cellpose_outlines(root_folder, 
                    cell_diameter=230,
                    nucleus_diameter=100,
                    cyto_channel = 2,
                    nucleus_channel = 3,
                    save_masks=True):

    masks_folder_name = os.path.join(root_folder, "masks", "")
    max_folder_name = os.path.join(root_folder, "max_proj_HELM", "")
    summary_file_name = os.path.join(root_folder, "summary.txt")
    cp_outlines_folder_name = os.path.join(root_folder, "outlines_cellpose", "")
    file_names = logging(root_folder)
    channels= [[cyto_channel, nucleus_channel]]

    for i, path in enumerate(file_names):
        print(path)
        tiff = rw.open_png(path)
        file_name = path.split("/")[-1]#TODO not / but OS separator

        # cytosol masks 
        masks, flows, styles, diams = get_cell_mask(
            tiff, channels = channels, diameter = cell_diameter)
        rw.save_mask_as_tiff(masks, masks_folder_name+file_name[:-4]+"_cyto_mask.tif")
        rw.save_outlines_as_text(masks, cp_outlines_folder_name+file_name[:-4]+"_cytosol")

        # nuclear masks
        masks, flows, styles, diams = get_nuclear_mask(tiff, diameter=nucleus_diameter)
        rw.save_mask_as_tiff(masks, masks_folder_name+file_name[:-4]+"_nuclear_mask.tif")
        rw.save_outlines_as_text(masks, cp_outlines_folder_name+file_name[:-4]+"_nuclear")

        # log
        with open(summary_file_name, 'r') as log_file:
            lines = log_file.readlines()
        lines[i] = path+",1\n"
        with open(summary_file_name, 'w+') as new_file:
            for line in lines:
                new_file.write(line)

def FQ_outlines_from_cp(root_folder, search_key = '*_cytosol_cp_outlines.txt'):

    cp_outlines_folder_name = os.path.join(root_folder, "outlines_cellpose", "")
    FQ_outlines_folder_name = os.path.join(root_folder, "outlines_FQ", "")

    for path in Path(root_folder).rglob(search_key):

        splited = path.name.split("_")
        IMG_raw = splited[0]+"_"+splited[1]+"_CH1.tif"
        IMG_DAPI = splited[0]+"_"+splited[1]+"_CH3.tif"

        cp_cyto_outline = os.path.join(cp_outlines_folder_name, path.name)
        cp_nuclei_outline = os.path.join(
            cp_outlines_folder_name, path.name.replace("cytosol", "nuclear"))
        
        header = construct_header(IMG_raw, IMG_DAPI)
        outlines, n = construct_cell_dic(cp_nuclei_outline, cp_cyto_outline)
        outfile_name = os.path.join(
            FQ_outlines_folder_name, splited[0]+"_"+splited[1]+"_FQ_outline.txt")
        dump(outfile_name, header, outlines, n)

def get_nuclear_mask(image, channels = [[3,0]], diameter = None):
    model = models.Cellpose(
        gpu=True,
        model_type='nuclei')
    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        resample=True,
        channels=channels)
    return masks, flows, styles, diams

def get_cell_mask(image, channels = [[2,3]], diameter = 230):
    model = models.Cellpose(
        gpu=True,
        model_type='cyto')
    masks, flows, styles, diams = model.eval(
        image,
        diameter=230,
        resample=True,
        channels=channels)
    return masks, flows, styles, diams

def construct_header(IMG_raw, IMG_DAPI):
    header = {}
    header['FISH-QUANT'] = 'v3a'
    header['File-version'] = '3D_v1'
    header['date'] = 'RESULTS OF SPOT DETECTION PERFORMED ON '
    header['IMG_Raw'] = IMG_raw
    header['IMG_Filtered'] = ''
    header['IMG_DAPI'] = IMG_DAPI
    header['IMG_TS_label'] = ''
    header['FILE_settings'] = ''
    header['Pix-XY'] = '160'
    header['Pix-Z'] = '300'
    header['RI'] = '1.33'
    header['Ex'] = '547'
    header['Em'] = '583'
    header['NA'] = '1.4'
    header['Type'] = 'widefield'
    return header

def construct_cell_dic(nuclei_outline, cyto_outline):
    """Construct dictionary entries for all the nucleus and cytoplasms
    outlines in the Cellpose output text file.
    
    WARNING: I don't know if the order of the masks are the same."""

    outlines_dic = {}
    read_cyto_file = glob.glob(cyto_outline)
    infile = open(cyto_outline, "r")
    lines = infile.readlines()
    numberOfCells = len(lines)
    for x in range(numberOfCells):
        key = f"Cell_{x+1}_"
        line = lines[x]
        coords = line.split(",")
        outlines_dic[key+"X"] = coords[::2]
        outlines_dic[key+"Y"] = coords[1::2]

    read_nucl_file = glob.glob(nuclei_outline)
    infile = open(nuclei_outline, "r")
    lines = infile.readlines()
    numberOfCells = len(lines)
    for x in range(numberOfCells):
        key = f"Nucleus_{x+1}_"
        line = lines[x]
        coords = line.split(",")
        outlines_dic[key+"X"] = coords[::2]
        outlines_dic[key+"Y"] = coords[1::2]
    return outlines_dic, numberOfCells

def to_string(dic_value):
    out_str = ""
    for coord in dic_value[:-1]:
        val = int(coord)
        out_str += str(val)+"\t"
    out_str += str(int(dic_value[-1]))+"\n"
    return out_str

def dump(outfile_name, header, outlines, numberOfCells):

    with open(outfile_name, "w+") as outfile:

        outfile.write(
            "FISH-QUANT\tv3a\nFile-version\t3D_v1\nRESULTS OF SPOT DETECTION PERFORMED ON 02-Jun-2021\n")
        outfile.write(
            "COMMENT\n"+
            "IMG_Raw\t"+header['IMG_Raw']+"\n"+
            "IMG_Filtered\t"+header['IMG_Filtered']+"\n"+
            "IMG_DAPI\t"+header['IMG_DAPI']+"\n"+
            "IMG_TS_label\t"+header['IMG_TS_label']+"\n"
        )
        outfile.write(
            "FILE_settings\nPARAMETERS\n"+
            "Pix-XY\tPix-Z\tRI\tEx\tEm\tNA\tType\n"+
            header['Pix-XY']+"\t"+
            header['Pix-Z']+"\t"+
            header['RI']+"\t"+
            header['Ex']+"\t"+
            header['Em']+"\t"+
            header['NA']+"\t"+
            header['Type']+"\n"
        )
        for i in range(numberOfCells):
            cell_key = f"Cell_{i+1}"
            X_POS = outlines[cell_key+'_X']
            X_POS = to_string(X_POS)
            Y_POS = outlines[cell_key+'_Y']
            Y_POS = to_string(Y_POS)

            nuc_key = f"Nucleus_{i+1}_"
            nuc_X_POS = outlines[nuc_key+'X']
            nuc_X_POS = to_string(nuc_X_POS)
            nuc_Y_POS = outlines[nuc_key+'Y']
            nuc_Y_POS = to_string(nuc_Y_POS)

            outfile.write(
                "CELL_START\t"+cell_key+"\n"+
                "X_POS\t"+X_POS+
                "Y_POS\t"+Y_POS+
                "Z_POS\t\nCELL_END\n"+
                f"Nucleus_START\tNUC_auto_{i+1}\n"+
                "X_POS\t"+nuc_X_POS+
                "Y_POS\t"+nuc_Y_POS+
                "Z_POS\t\nNucleus_END\n"
            )


if __name__ == "__main__":
    # open tiff, convert to array, get max z-projection
    dataset = Image.open("MAX_4OHT_21_1_DAPI.tif")
    h,w = np.shape(dataset)
    tiffarray = np.zeros((h,w))
    tiffarray[:,:] = np.array(dataset)

    masks, flows, styles, diams = get_nuclear_mask(tiffarray)
    np.savetxt("masks_nuclear.csv", masks, delimiter=',')
    tif.imsave("masks_nuclear.tif", masks, bigtiff=True)
    outlines = utils.outlines_list(masks)
    io.outlines_to_text("mask_nuclei", outlines)

    # cellprob = flows[2]


    file_path = '4OHT_21_1_CH1.tif'
    tiffarray = rw.open_tiff(file_path)
    tiffarray = tiffarray.astype('uint16')

    # save the max proj as tiff



    # get the mean nuclear intensity in p53 max proj within
    # all masks.  

    dataset_p53 = Image.open("MAX_4OHT_21_1_p53.tif")
    h,w = np.shape(dataset_p53)
    p53_array = np.zeros((h,w))
    p53_array[:,:] = np.array(dataset_p53)

    out_name = 'MAX_PYTHON_21_DAPI.tif'
    maxim = rw.maxproj(tiffarray)
    print(np.sum(maxim))
    rw.save_stack_as_tiff(maxim, out_name)

    # get masks with cellpose for all max projections

    for n in range(1,np.max(masks)+1):
        rows, cols = np.where(masks==n)
        print(np.mean(p53_array[rows, cols]))
    file_path = 'MAX_4OHT_21.tif'
    tiffarray = rw.open_tiff(file_path)
    masks, flows, styles, diams = get_cell_mask(tiffarray)
    np.savetxt("masks_cyto.csv", masks, delimiter=',')
    tif.imsave("masks_cyto.tif", masks, bigtiff=True)
    outlines = utils.outlines_list(masks)
    io.outlines_to_text("mask_cyto", outlines)
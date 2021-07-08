import os
import glob
import numpy as np
from pathlib import Path

# I think the whole thing could be done faster with the masks 
# and np.where(masks == value), check what value is inside those index
# on the nuclear masks, assign each nucleus and then exclude based 
# on criterion. 

def cp_outline_to_numpy(filename):
    infile = open(filename)
    cyto_lines = infile.readlines()
    infile.close()
    all_coords = []
    for line in cyto_lines:
        splitted = line.split(",")
        coord = np.zeros(((len(splitted)//2),2), dtype='int')
        coord[:,0]=splitted[::2]
        coord[:,1]=splitted[1::2]
        all_coords.append(coord)
    return all_coords

def PolyArea(x,y):
    # Shoelace method from https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def validate_size(coords, valid_cells):
    """Marks as unvalid cells that are less than 20 times 
    smaller than the mean size of cells in that image."""

    areas = []
    # calculate all cell area to get the mean.
    for i, cell in enumerate(coords):
        if valid_cells[i]==0:#ignore unvalid cells
            continue
        area = PolyArea(cell[:,0], cell[:,1])
        areas.append(area)
    mean_area = np.mean(areas)

    for i, a in enumerate(areas):
        if a<mean_area/20:
            valid_cells[i]=0
            print(f"Cell {i+1} "
                "is +20x smaller than the rest and was excluded")
    return valid_cells

def reject_border_cells(cyto_coords, valid_cells):
    """Marks as invalid cells that have more than 10% of
    their outline on the border of the image."""
    for i, cell in enumerate(cyto_coords):
        #TODO: this shouldn't 1024 hardcoded
        xs, ys = np.where((cell==0) | (cell == 1024))
        if len(xs)>0.1*len(cell):
            valid_cells[i]=0
            print(f"Cell {i+1} "
                "is +10% on border and was excluded.")
    return valid_cells

def validate_nucleus_assignment(cyto_coords, nuc_coords, valid_cells):
    """
    Marks which nucleus (by index) is inside which cell. Cells
    without a nucleus inside are marked invalid, and nucleus
    without cells also.
    Check via the bounding boxes of cytosol and nucleus.
    """
    # -1 means not assigned to a cell
    assigned_nuclei=np.zeros(len(nuc_coords))-1
    for i, cell in enumerate(cyto_coords):
        if valid_cells[i]==0:#ignore invalid cells
            continue

        has_nucleus = False
        for j, nuc in enumerate(nuc_coords):
            if assigned_nuclei[j] != -1:#nucleus is already assigned
                continue

            if check_if_contains(cell, nuc):
                assigned_nuclei[j]=i
                has_nucleus = True
                break
        
        if not has_nucleus:
            valid_cells[i]=0
            print(f"Cell {i+1} "
                "has no nucleus inside and was excluded.")
    return assigned_nuclei, valid_cells

def rewrite_outline_files(
    filename, valid_cells, assigned_nuclei):
    """
    Rewrites the cp_outline_files so that the cytosol file contains only valid cells
    and the nuclear file contains a nucleus for each, at the corresponding line.
    """
    cyto_outline_filename = filename
    nuc_outline_filename = filename.replace("cytosol", "nuclear")

    with open(cyto_outline_filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        for i, cell in enumerate(valid_cells):
            if cell!= 0 and cell!=1:
                raise AssertionError("Something went wrong and the valid_cells array is corrupted.")
            if cell==0:
                continue
            if cell==1:
                f.write(lines[i])
        f.truncate()

    with open(nuc_outline_filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        for i, cell in enumerate(valid_cells):
            if cell == 1:#TODO:should be a boolean directly
                for j, assignment in enumerate(assigned_nuclei):
                    if assignment == i:
                        f.write(lines[j])
        f.truncate()


def check_if_contains(cell, nucleus):
    """Simple axis-aligned bounding box check"""
    cell_max=np.max(cell, axis=0)
    cell_min=np.min(cell, axis=0)
    nuc_max=np.max(nucleus, axis=0)
    nuc_min=np.min(nucleus, axis=0)

    if nuc_max[0]>cell_max[0] or nuc_max[1]>cell_max[1]:
        return False
    if nuc_min[0]<cell_min[0] or nuc_min[1]<cell_min[1]:
        return False
    return True
    
def batch_sanity_check(root_folder,
                    exclude_border=True,
                    exclude_too_small=True,
                    reassign_nucleus=True):
    """
    Performs the specified checks of the cells and nucleus
    detected by Cellpose to reduce amount of attribution errors.
    """
    for path in Path(root_folder).rglob('*cytosol_cp_outlines.txt'):
        print(path.name)
        sanity_check(
            path,
            exclude_border=exclude_border,
            exclude_too_small=exclude_too_small,
            reassign_nucleus=reassign_nucleus)

def sanity_check(filename,
                exclude_border=True,
                exclude_too_small=True,
                reassign_nucleus=True):
    """Single sanity check"""

    cyto_outline_filename = os.path.abspath(filename)
    nuc_outline_filename = cyto_outline_filename.replace("cytosol", "nuclear")

    # if root folder doesn't ends with /, add it
    # TODO: join so that it works on windows too.

    try:
        cyto_coords = cp_outline_to_numpy(cyto_outline_filename)
        nuc_coords = cp_outline_to_numpy(nuc_outline_filename)
    except FileNotFoundError : 
        raise FileNotFoundError(
            "Please make sure Cellpose cell outlines end with "
            "cytosol_cp_outlines.txt, and nuclear outlines with "
            "nuclear_cp_outlines.txt, and are in the root folder "
            "you provided.")
    
    valid_cells = np.ones(len(cyto_coords))
    assigned_nuclei=np.zeros(len(nuc_coords))-1

    if exclude_too_small:
        valid_cells = validate_size(cyto_coords, valid_cells)                

    if exclude_border:
        valid_cells = reject_border_cells(cyto_coords, valid_cells)
    
    if reassign_nucleus:
        assigned_nuclei, valid_cells = \
            validate_nucleus_assignment(cyto_coords, nuc_coords, valid_cells)

    rewrite_outline_files(cyto_outline_filename, valid_cells, assigned_nuclei)

if __name__ == "__main__":
    root_folder = "/home/tom/Documents/Scientifique/Thesis/Data/EXP264/nutlin_max/"
    batch_sanity_check(root_folder)
    #TODO: This shouldn't overwrite user data before making sure we are correct.
    saving_folder_max = "/home/tom/Documents/Scientifique/Thesis/Data/EXP264/nutlin_max/"
    import cellpose_to_FISHQuant as FQ
    for path in Path(root_folder).rglob('*cytosol_cp_outlines.txt'):
        # generate the outline files for FISHQuant
        print(path.name)
        splited = path.name.split("_")
        IMG_raw = splited[0]+"_"+splited[1]+"_CH1.tif"
        IMG_DAPI = splited[0]+"_"+splited[1]+"_CH3.tif"
        cp_nuclei_outline = saving_folder_max+path.name
        cp_cyto_outline = saving_folder_max+path.name.replace("cytosol", "nuclear")
        header = FQ.construct_header(IMG_raw, IMG_DAPI)
        outlines, n = FQ.construct_cell_dic(cp_nuclei_outline, cp_cyto_outline)
        outfile_name = saving_folder_max+IMG_raw[:-4]+"_FQ_outline.txt"
        FQ.dump(outfile_name, header, outlines, n)
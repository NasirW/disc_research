import os
import numpy as np
import pandas as pd
import ast
import glob
from PIL import Image
import pickle
from tqdm import tqdm

# Constants
BIOP, SUSP  = "biop",  "susp"
RES, SEN    = "res",   "sen"
BEN, MAL    = "ben",   "mal"
FILES, DATA, DIR, PKL = "files", "data", "data_dir", "pickle"
CSV, HIST, CORR, MSE = "csv", "hist", "corr", "mse"

BOX_SAVE = os.path.expanduser("~/Library/CloudStorage/Box-Box/DISC_StudyGrp_DLChemoResistance/saved_plots")

GENERAL_SAVE  = os.path.join(BOX_SAVE, "general")
BIOP_RES_SAVE = os.path.join(BOX_SAVE, "random_biop_res")
BIOP_SEN_SAVE = os.path.join(BOX_SAVE, "random_biop_sen")
META_BEN_SAVE = os.path.join(BOX_SAVE, "random_meta_ben")
META_MAL_SAVE = os.path.join(BOX_SAVE, "random_meta_mal")

os.makedirs(os.path.join(GENERAL_SAVE, CSV),  exist_ok=True)
os.makedirs(os.path.join(GENERAL_SAVE, HIST), exist_ok=True)
os.makedirs(os.path.join(GENERAL_SAVE, CORR), exist_ok=True)
os.makedirs(os.path.join(GENERAL_SAVE, MSE),  exist_ok=True)


# File paths
chemo_res = { BIOP: {DIR: os.path.join('data', 'chemo_res_biop')},
              SUSP: {DIR: os.path.join('data', 'chemo_res_susp')} }

for DATASET in chemo_res.values():
    DATASET[RES] = {FILES: glob.glob(os.path.join(DATASET[DIR], 'Norm_Resistant', '*.csv')),
                    DATA: None}
    DATASET[SEN] = {FILES: glob.glob(os.path.join(DATASET[DIR], 'Norm_Sensitive', '*.csv')),
                    DATA: None}
    del DATASET[DIR]

metastasis = { BEN: {FILES: glob.glob(os.path.join('data', 'metastasis', 'Classification', 'Benign', '*.csv')),
                     DATA: None},
               MAL: {FILES: glob.glob(os.path.join('data', 'metastasis', 'Classification', 'Malignant', '*.csv')),
                     DATA: None} }

metastasis_dir = os.path.join('data', 'metastasis')


def split_chemo_filename(filename):
    _, patient_lesion, capture = os.path.basename(filename).split('_')
    patient_num, lesion_num = patient_lesion.split('.')
    capture_num, _ = capture.split('.')
    return patient_num, lesion_num, capture_num

def split_chemo_all_filename(filename):
    _, patient_num, lesion_num, capture = os.path.basename(filename).split('_')
    capture_num, _ = capture.split('.')
    return patient_num, lesion_num, capture_num

def split_patch_filename(filename):
    _, _, patient_lesion, capture = os.path.basename(filename).split('_')
    patient_num, lesion_num = patient_lesion.split('.')
    capture_num, _ = capture.split('.')
    return patient_num, lesion_num, capture_num

def split_filename(filename):
    if "ChemoAll" in filename:
        return split_chemo_all_filename(filename)
    if "Patch" in filename:
        return split_patch_filename(filename)
    return split_chemo_filename(filename)

def img_to_arr(file_name, new_csv_size=(80, 80, 3)):
    ext = os.path.splitext(file_name)[1]
    
    match ext:
        case '.csv':
            img = pd.read_csv(file_name, header=None)
            return img.to_numpy().reshape(new_csv_size)
        case '.png':
            img = np.array(Image.open(file_name), dtype=np.uint16)
            return img
        case '.pkl':
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        case _:
            raise ValueError(f"Unsupported file extension: {ext}")
        
def replace_ext(file_name, new_ext):
    return os.path.splitext(file_name)[0] + new_ext

def add_folder_to_path(old_path, new_folder, index):
    folders = old_path.split(os.path.sep)
    folders.insert(index, new_folder)
    return os.path.join(*folders)

    
def save_imgs_to_pkl(dataset):
    for subset in dataset.values():
        for filename in tqdm(subset[FILES]):
            img = img_to_arr(filename)
            
            pkl_path = add_folder_to_path(replace_ext(filename, '.pkl'), PKL, 1)
            
            if os.path.exists(pkl_path):
                continue

            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(img, f)
                
    return dataset

def setup_dataset(dataset, pickled=False):
    for subset in tqdm(dataset.values()):
        patients = []
        captures = []
        lesions = []
        images = []

        for filename in subset[FILES]:
            patient_num, lesion_num, capture_num = split_filename(filename)
            
            patients.append(patient_num)
            captures.append(capture_num)
            lesions.append(lesion_num)
            
            if pickled:
                filename = replace_ext(filename, '.pkl')
            
            images.append(img_to_arr(filename))
            
        subset[DATA] = pd.DataFrame({'Patient Number': patients, 'Capture Number': captures, 'Lesion Number': lesions, 'Image': images})
    return dataset

for dataset in [chemo_res[BIOP], chemo_res[SUSP], metastasis]:
    dataset = save_imgs_to_pkl(dataset)
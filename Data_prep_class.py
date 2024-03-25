import cv2
import numpy as np
import os
from os.path import join as path_join
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm
from argparse import ArgumentParser as arg_parser
from colorama import Fore as F


class Path:
    BIOP, SUSP = "biop",  "susp"
    RES, SEN = "res",   "sen"
    BEN, MAL = "ben",   "mal"
    FILES, DATA, DIR, PKL = "files", "data", "data_dir", "pickle"
    CSV, HIST, CORR, MSE = "csv", "hist", "corr", "mse"

    def split_filename_type(filename, type):
        match type:
            case "chemo":
                _, patient_lesion, capture = os.path.basename(
                    filename).split('_')
                patient_num, lesion_num = patient_lesion.split('.')
                capture_num, _ = capture.split('.')
                return int(patient_num), lesion_num, int(capture_num)

            case "chemo_all":
                _, patient_num, lesion_num, capture = os.path.basename(
                    filename).split('_')
                capture_num, _ = capture.split('.')
                return int(patient_num), lesion_num, int(capture_num)

            case "patch":
                _, _, patient_lesion, capture = os.path.basename(
                    filename).split('_')
                patient_num, lesion_num = patient_lesion.split('.')
                capture_num, _ = capture.split('.')
                return int(patient_num), lesion_num, int(capture_num)

    def split_filename(filename):
        if "ChemoAll" in filename:
            return Path.split_filename_type(filename, "chemo_all")
        if "Patch" in filename:
            return Path.split_filename_type(filename, "patch")
        return Path.split_filename_type(filename, "chemo")

    def replace_ext(file_name, new_ext):
        return os.path.splitext(file_name)[0] + new_ext

    def add_folder_to_path(old_path, new_folder, index):
        folders = old_path.split(os.path.sep)
        folders.insert(index, new_folder)
        return os.path.join(*folders)


class ImageLoader:
    def __init__(self, resistant_folder, sensitive_folder, name):
        self.data = None
        self.name = name
        
        self.dataset = {Path.RES: {Path.FILES: glob(os.path.join(resistant_folder, '*')),
                                   Path.DATA:  None,
                                   Path.DIR:   resistant_folder},
                        Path.SEN: {Path.FILES: glob(os.path.join(sensitive_folder, '*')),
                                   Path.DATA:  None,
                                   Path.DIR:   sensitive_folder}}
        
        self.setup_dataset()
        
        
    @staticmethod
    def _img_to_arr(file_name, new_csv_size=(80, 80, 3)):
        ext = os.path.splitext(file_name)[1]

        match ext:
            case '.csv':
                img = pd.read_csv(file_name, header=None)
                return np.stack(img.to_numpy().reshape(new_csv_size))
            case '.png':
                img = np.array(Image.open(file_name), dtype=np.uint16)
                return img
            case '.pkl':
                with open(file_name, 'rb') as f:
                    return np.stack(pickle.load(f))
            case _:
                raise ValueError(f"Unsupported file extension: {ext}")
            
    def pickle_imgs(self, overwrite=False):
        for subset in self.dataset.values():
            for filename in tqdm(subset[Path.FILES]):
                img = ImageLoader._img_to_arr(filename)
                
                pkl_path = Path.add_folder_to_path(Path.replace_ext(filename, '.pkl'), Path.PKL, 1)
                
                if not overwrite and os.path.exists(pkl_path):
                    continue

                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(img, f)
                    
    def setup_dataset(self, pickled=True):
        for type, subset in self.dataset.items():
            patients = []
            captures = []
            lesions = []
            images = []

            for filename in tqdm(subset[Path.FILES]):
                patient_num, lesion_num, capture_num = Path.split_filename(filename)

                patients.append(patient_num)
                captures.append(capture_num)
                lesions.append(lesion_num)

                pkl_path = Path.add_folder_to_path(
                    Path.replace_ext(filename, '.pkl'), Path.PKL, 1)

                if pickled and os.path.exists(pkl_path):
                    filename = pkl_path

                images.append(np.stack(ImageLoader._img_to_arr(filename)))

            labels = [1 if type in [Path.RES, Path.MAL] else 0] * len(images)

            multi_index = pd.MultiIndex.from_arrays([patients, lesions, captures],
                                                    names=('Patient Number', 'Lesion Number', 'Capture Number'))
            subset[Path.DATA] = pd.DataFrame({'Image': images, 'Label': labels}, index=multi_index)
            subset[Path.DATA].sort_index(inplace=True)
            
        self.data = pd.concat([self.dataset[Path.RES][Path.DATA], self.dataset[Path.SEN][Path.DATA]])

    def split_data(self, test_prop=0.2, val_prop=0.15, random_state=42):
        X = self.data['Image']
        y = self.data['Label']
        
        assert len(X) == len(y), "Length of images and labels do not match"
        
        val_size  = int(len(X) * val_prop)
        test_size = int(len(X) * test_prop)
        
        X_train_val, X_test, y_train_val, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
            
        X_train, X_val, y_train, y_val = \
            train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)
            

        self.train_images = X_train.to_numpy()
        self.train_labels = y_train.to_numpy()
        self.val_images   = X_val.to_numpy()
        self.val_labels   = y_val.to_numpy()
        self.test_images  = X_test.to_numpy()
        self.test_labels  = y_test.to_numpy()
        
        return self.train_images, self.train_labels, \
               self.val_images,   self.val_labels, \
               self.test_images,  self.test_labels
        
    def save_splits(self, save_dir):
        save_dir = path_join(save_dir, self.name)
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'X_train.npy'), self.train_images)
        np.save(os.path.join(save_dir, 'y_train.npy'), self.train_labels)
        np.save(os.path.join(save_dir, 'X_val.npy'),   self.val_images)
        np.save(os.path.join(save_dir, 'y_val.npy'),   self.val_labels)
        np.save(os.path.join(save_dir, 'X_test.npy'),  self.test_images)
        np.save(os.path.join(save_dir, 'y_test.npy'),  self.test_labels)
        
        print(F.GREEN + f"\nSaved `{self.name}` Data to: {save_dir}" + F.RESET, end='\n\n')
        

if __name__ == '__main__':
    parser = arg_parser()

    parser.add_argument("--res_folder", default=path_join("data", "chemo_res_biop", "Norm_Resistant"),
                        help="The folder containing the resistant images", type=str)
    
    parser.add_argument("--sen_folder", default=path_join("data", "chemo_res_biop", "Norm_Sensitive"),
                        help="The folder containing the sensitive images", type=str)
    
    parser.add_argument("--name", default="chemo_res_biop",
                        help="The name of the dataset", type=str)
    
    parser.add_argument("--pickle", default="n",
                        help="To pickle the images, select from: ['n' -> No pickling, 'ow' -> Pickling with overwrite, 'n-ow' -> Pickling without overwrite]",
                        choices=["n", "ow", "n-ow"])
                        
    args = parser.parse_args()
    
    data_loader = ImageLoader(resistant_folder=args.res_folder, 
                              sensitive_folder=args.sen_folder,
                              name=args.name)

    data_loader.split_data()
    
    if args.pickle != "n":
        data_loader.pickle_imgs(overwrite=args.pickle == "ow")
        
    data_loader.save_splits(save_dir="results")
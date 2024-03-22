import cv2
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm


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
                return patient_num, lesion_num, capture_num

            case "chemo_all":
                _, patient_num, lesion_num, capture = os.path.basename(
                    filename).split('_')
                capture_num, _ = capture.split('.')
                return patient_num, lesion_num, capture_num

            case "patch":
                _, _, patient_lesion, capture = os.path.basename(
                    filename).split('_')
                patient_num, lesion_num = patient_lesion.split('.')
                capture_num, _ = capture.split('.')
                return patient_num, lesion_num, capture_num

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
    def __init__(self, resistant_folder, sensitive_folder):
        self.dataset = {Path.RES: {Path.FILES: glob(os.path.join(resistant_folder, '*')),
                                   Path.DATA:  None,
                                   Path.DIR:   resistant_folder},
                        Path.SEN: {Path.FILES: glob(os.path.join(sensitive_folder, '*')),
                                   Path.DATA:  None,
                                   Path.DIR:   sensitive_folder}}
        
        

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

            subset[Path.DATA] = pd.DataFrame({'Patient Number': patients,
                                              'Capture Number': captures,
                                              'Lesion Number': lesions,
                                              'Image': images,
                                              'Label': labels})

    def prepare_data(self):
        resistant_images = np.stack(
            self.dataset[Path.RES][Path.DATA]['Image'].to_numpy())
        sensitive_images = np.stack(
            self.dataset[Path.SEN][Path.DATA]['Image'].to_numpy())
        resistant_labels = np.stack(
            self.dataset[Path.RES][Path.DATA]['Label'].to_numpy())
        sensitive_labels = np.stack(
            self.dataset[Path.SEN][Path.DATA]['Label'].to_numpy())

        self.images = np.concatenate((resistant_images, sensitive_images))
        self.labels = np.concatenate((resistant_labels, sensitive_labels))
        self.info   = list(zip(self.dataset[Path.RES][Path.DATA]['Patient Number'].to_numpy(),
                               self.dataset[Path.RES][Path.DATA]['Capture Number'].to_numpy()))


    def split_data(self, test_size=0.1, random_state=42):
        self.prepare_data()
        grouped_list = self.group_images(self.images, self.labels, self.info)
        grouped_list = list(grouped_list.values())
        np.random.shuffle(grouped_list)
        train, test = train_test_split(
            grouped_list, test_size=test_size, random_state=random_state)
        self.train_images, self.train_labels = self.combine_data(train)
        self.test_images, self.test_labels = self.combine_data(test)

    def group_images(self, images, labels, info):
        grouped_data = {}
        for img, label, info_tuple in zip(images, labels, info):
            patient_lesion_id, replicate = info_tuple
            key = patient_lesion_id.split('.')[0]
            if key not in grouped_data:
                grouped_data[key] = {'images': [], 'labels': [], 'info': []}
            grouped_data[key]['images'].append(img)
            grouped_data[key]['labels'].append(label)
            grouped_data[key]['info'].append((patient_lesion_id, replicate))
        return grouped_data

    def combine_data(self, groups):
        combined_images = []
        combined_labels = []
        for group in groups:
            combined_images.extend(group['images'])
            combined_labels.extend(group['labels'])
        return np.array(combined_images), np.array(combined_labels)

    def create_dataframe(self, groups, group_name):
        data = []
        for group in groups:
            for patient_lesion_id, _ in group['info']:
                patient_id, lesion_id = patient_lesion_id.split('.')
                data.append({'Patient_ID': patient_id,
                            'Lesion_ID': lesion_id, 'Group': group_name})
        return pd.DataFrame(data)

    def save_data_as_npy(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'class_X_train.npy'), self.train_images)
        np.save(os.path.join(save_path, 'class_y_train.npy'), self.train_labels)
        np.save(os.path.join(save_path, 'class_X_test.npy'),  self.test_images)
        np.save(os.path.join(save_path, 'class_y_test.npy'),  self.test_labels)
        print(f"Saved data to {save_path}")


resistant_folder = 'data/chemo_res_biop/Norm_Resistant'
sensitive_folder = 'data/chemo_res_biop/Norm_Sensitive'
data_loader = ImageLoader(resistant_folder, sensitive_folder)

data_loader.setup_dataset()
data_loader.prepare_data()
data_loader.split_data()
data_loader.save_data_as_npy('results/chemo_res_biop')
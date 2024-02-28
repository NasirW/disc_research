import cv2
import numpy as np
import os
import re


pattern = re.compile(r'^Chemo_(\d+\.\d+)_(\d+)\.png$')

def load_images_from_folder(folder, label):
    images = []
    labels = []
    info = []  # To store patient and lesion IDs

    file_names = os.listdir(folder)

    # Get total number of files for progress tracking
    total_files = len(file_names)
    print(f"Total files found: {total_files}")
    #file_count = 0

    for index, filename in enumerate(file_names, start=1):
        if index % 100 == 0:  # Progress update every 100 images
            print(f"Processing file {index}/{total_files}")

        # This regex pattern accounts for 'ChemoAll' or 'Chemo' and patient IDs with a period
        #pattern = re.compile(r'Chemo(All_)?(\d+(?:\.\d+)?)_(\d+)_(\d+)\.png')
        match = pattern.match(filename)
        if match:
            patient_lesion_id, replicate = match.groups()
            #print(f"Extracted IDs - Patient: {patient_id}, Lesion: {lesion_id}")
            # Load and preprocess the image, then store the information
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype('float32') / 255
                images.append(img_normalized)
                labels.append(label)
                info.append((patient_lesion_id, replicate))
            else:
                print(f"Filename does not match pattern: {filename}")
         # Diagnostic print for info
        #print(f"Total info entries: {len(info)}")
        #print(f"Sample info entries: {info[:5]}")  # Print first 5 entries for checking
    # After loading the images and labels...
    print(f"Number of images: {len(images)}")
    print(f"Number of labels: {len(labels)}")

    if len(images) != len(labels):
        print("Mismatch detected! The number of images and labels are not equal.")
        # Optional: Implement additional debugging steps here to find out which images/labels are causing the mismatch.


    return images, labels, info

# Paths to image folders
resistant_folder = '../DISC_StudyGrp_DLChemoResistance/Data_Chemotherapy Resistance/Biopsied Lesions/Resistant'
sensitive_folder = '../DISC_StudyGrp_DLChemoResistance/Data_Chemotherapy Resistance/Biopsied Lesions/Sensitive'

# Assuming 'Resistant' maps to 0 and 'Sensitive' maps to 1
resistant_images, resistant_labels, resistant_info = load_images_from_folder(resistant_folder, 0)
sensitive_images, sensitive_labels, sensitive_info = load_images_from_folder(sensitive_folder, 1)

# Combine datasets
all_images = np.array(resistant_images + sensitive_images)
all_labels = np.array(resistant_labels + sensitive_labels)
all_info = resistant_info + sensitive_info

from sklearn.model_selection import train_test_split


def group_images(images, labels, info):
    grouped_data = {}
    for img, label, info_tuple in zip(images, labels, info):
        patient_lesion_id, replicate = info_tuple  # Correctly unpack the tuple
        patient_id = patient_lesion_id.split('.')[0]  # Extract only the patient ID
        key = patient_id
        if key not in grouped_data:
            grouped_data[key] = {'images': [], 'labels': [], 'info': []}
        grouped_data[key]['images'].append(img)
        grouped_data[key]['labels'].append(label)
        # Optionally store the full patient_lesion_id and replicate for completeness
        grouped_data[key]['info'].append((patient_lesion_id, replicate))
    return grouped_data

# Group images
grouped_images = group_images(all_images, all_labels, all_info)

print(grouped_images)
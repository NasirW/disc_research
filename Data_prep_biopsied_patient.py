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
    for img, label, patient_lesion_id in zip(images, labels, info):
        patient_id = patient_lesion_id.split('.')[0]  # Extract only the patient ID
        key = patient_id  # Use patient ID as the key for grouping

        if key not in grouped_data:
            grouped_data[key] = {'images': [], 'labels': [], 'info': []}
        grouped_data[key]['images'].append(img)
        grouped_data[key]['labels'].append(label)
        grouped_data[key]['info'].append(patient_lesion_id)  # Store the full patient_lesion_id for potential reference
    return grouped_data

# Group images
grouped_images = group_images(all_images, all_labels, all_info)

# Convert grouped data into a list for splitting
grouped_list = list(grouped_images.values())

# Shuffle the grouped list to randomize data
np.random.shuffle(grouped_list)

# Check the size of the grouped data
print(f"Total number of groups: {len(grouped_list)}")

# Splitting the data into training, validation, and test sets
#train_val, test = train_test_split(grouped_list, test_size=0.05, random_state=42)
#train, val = train_test_split(train_val, test_size=0.05 / 0.95, random_state=42)
train, test = train_test_split(grouped_list, test_size=0.4, random_state=42)


# Function to combine images and labels from grouped data
def combine_data(groups):
    combined_images = []
    combined_labels = []
    for group in groups:
        combined_images.extend(group['images'])
        combined_labels.extend(group['labels'])
    return np.array(combined_images), np.array(combined_labels)

# Combine images and labels for each set
train_images, train_labels = combine_data(train)
#val_images, val_labels = combine_data(val)
test_images, test_labels = combine_data(test)

import pandas as pd

# Function to create a DataFrame from grouped data
def create_dataframe(groups, group_name):
    data = []
    for group in groups:
        for patient_lesion_id in group['info']:
            patient_id, lesion_id = patient_lesion_id.split('.')  # Splitting based on your input
            data.append({'Patient_ID': patient_id, 'Lesion_ID': lesion_id, 'Group': group_name})
    return pd.DataFrame(data)

# Create DataFrames for each set
train_df = create_dataframe(train, 'Training')
#val_df = create_dataframe(val, 'Validation')
test_df = create_dataframe(test, 'Testing')

# Combine all DataFrames
#all_df = pd.concat([train_df, val_df, test_df])

all_df = pd.concat([train_df, test_df])

# Export to CSV
all_df.to_csv('../disc_research_images/gen/biopsied/6040/patient_lesion_groups.csv', index=False)

# Saving the image datasets and labels in standard format
X_train, Y_train = train_images, train_labels
#X_val, Y_val = val_images, val_labels
X_test, Y_test = test_images, test_labels

# Check the cardinality of each set
print(f"Training set: {len(X_train)} images, {len(Y_train)} labels")
#print(f"Validation set: {len(X_val)} images, {len(Y_val)} labels")
print(f"Testing set: {len(X_test)} images, {len(Y_test)} labels")

# Assert to confirm that the splits have equal numbers of images and labels
assert len(X_train) == len(Y_train), "Training set images and labels count mismatch!"
#assert len(X_val) == len(Y_val), "Validation set images and labels count mismatch!"
assert len(X_test) == len(Y_test), "Testing set images and labels count mismatch!"



np.save('../disc_research_images/gen/biopsied/6040/X_train.npy', X_train)
np.save('../disc_research_images/gen/biopsied/6040/y_train.npy', Y_train)
#np.save('../disc_research_images/gen/suspicious/X_val.npy', X_val)
#np.save('../disc_research_images/gen/suspicious/Y_val.npy', Y_val)
np.save('../disc_research_images/gen/biopsied/6040/X_test.npy', X_test)
np.save('../disc_research_images/gen/biopsied/6040/y_test.npy', Y_test)


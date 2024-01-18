import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    info = []
    files = os.listdir(folder_path)
    total_files = len(files)

    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".png"):  # assuming the image format is .png
            # Extract information from filename
            parts = filename.split('_')
            patient_id = parts[2].split('.')[0]
            lesion_id = parts[2].split('.')[1]
            replicate_id = parts[3].split('.')[0]
            
            # Load and preprocess the image
            image = load_and_preprocess_image(file_path)
            
            # Append to lists
            images.append(image)
            labels.append(label)
            info.append((patient_id, lesion_id, replicate_id))

        # Print progress
        if (i+1) % 100 == 0 or i+1 == total_files:  # update every 100 images or when done
            print(f'Loaded {i+1}/{total_files} images from {folder_path}')

    return images, labels, info


def load_and_preprocess_image(file_path):
    # Load image in BGR format
    image = cv2.imread(file_path)

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to 80x80 if not already
    if image_rgb.shape[0] != 80 or image_rgb.shape[1] != 80:
        image_rgb = cv2.resize(image_rgb, (80, 80))

    # Normalize pixel values
    image_normalized = image_rgb / 255.0

    return image_normalized

# Paths to the image folders
benign_folder_path = './../DISC_StudyGrp_DLChemoResistance/Data_Metastasis Classification/Benign'
metastasis_folder_path = './../DISC_StudyGrp_DLChemoResistance/Data_Metastasis Classification/Metastasis'

# Load benign and metastasis images
benign_images, benign_labels, benign_info = load_images_from_folder(benign_folder_path, 0)  # 0 for benign

metastasis_images, metastasis_labels, metastasis_info = load_images_from_folder(metastasis_folder_path, 1)  # 1 for metastasis




# Combine datasets
all_images = benign_images + metastasis_images
all_labels = benign_labels + metastasis_labels
all_info = benign_info + metastasis_info

# Convert lists to numpy arrays for machine learning processing
all_images_np = np.array(all_images)
all_labels_np = np.array(all_labels)
all_info_np = np.array(all_info, dtype=object)


# Function to split data while keeping patient data together
def patient_wise_split(info_np, images_np, labels_np, test_size=0.1, random_state=42):
    # Extract unique patient IDs
    unique_patients = np.unique(info_np[:, 0])
    
    # Split patient IDs into training and test sets
    patients_train, patients_temp = train_test_split(unique_patients, test_size=test_size, random_state=random_state)
    
    # Split patients_temp further into validation and test sets
    patients_val, patients_test = train_test_split(patients_temp, test_size=0.5, random_state=random_state)

    # Function to get indices for a set of patients
    def get_indices_for_patients(patients, info_np):
        indices = [i for i, info in enumerate(info_np) if info[0] in patients]
        return indices

    # Get indices for training, validation, and test sets
    train_indices = get_indices_for_patients(patients_train, info_np)
    val_indices = get_indices_for_patients(patients_val, info_np)
    test_indices = get_indices_for_patients(patients_test, info_np)

    # Split the data according to indices
    X_train, y_train = images_np[train_indices], labels_np[train_indices]
    X_val, y_val = images_np[val_indices], labels_np[val_indices]
    X_test, y_test = images_np[test_indices], labels_np[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

# Use the function to split the data
X_train, X_val, X_test, y_train, y_val, y_test = patient_wise_split(all_info_np, all_images_np, all_labels_np)



np.save('../disc_research_images/gen/meta/X_train.npy',X_train)
np.save('../disc_research_images/gen/meta/X_val.npy',X_val)

np.save('../disc_research_images/gen/meta/X_test.npy',X_test)

np.save('../disc_research_images/gen/meta/y_train.npy',y_train)
np.save('../disc_research_images/gen/meta/y_val.npy',y_val)

np.save('../disc_research_images/gen/meta/y_test.npy',y_test)
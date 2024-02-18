# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Georgios Georgalis
# """
# from PIL import Image
# import numpy as np
# import pandas as pd

# from keras.preprocessing import image
# import os

# from sklearn.model_selection import train_test_split
# import cv2
# import csv


# #%% Data_prep.py loads the image files and exports the appropriate data structures for training
# img_folder_sensitive = './../DISC_StudyGrp_DLChemoResistance/Data_Chemotherapy Resistance/Suspecious Lesions/Sensitive' #Folder that contains the sensitive biopsied RGB images
# img_folder_resistant = './../DISC_StudyGrp_DLChemoResistance/Data_Chemotherapy Resistance/Suspecious Lesions/Resistant' #Folder that contains the resistant biopsied RGB images
# #edge_folder_sensitive_A ='./../../Images_Biopsied/Sensitive_Canny_0.4'
# #edge_folder_resistant_A ='./../../Images_Biopsied/Resistant_Canny_0.4'
# #Path to .csv with the labels (0 = sensitive, 1 = resistant)

# #Image dimensions
# image_height = 80
# image_width = 80


# # load filenames
# img_filenames_sensitive = np.array(sorted(os.listdir(img_folder_sensitive)))#sort
# img_filenames_resistant = np.array(sorted(os.listdir(img_folder_resistant)))#sort

# d1 = np.transpose(np.vstack([img_filenames_sensitive, np.zeros(np.size(img_filenames_sensitive))]))
# d2 = np.transpose(np.vstack([img_filenames_resistant, np.ones(np.size(img_filenames_resistant))]))
# d = np.vstack([d1, d2])

# labels = pd.DataFrame(data = d, columns = ['image_name', 'label'])


# wells = labels['image_name']
# Nimages = np.size(wells) #Number of images

# #Empty arrays for RGB images (X)
# X = np.zeros(shape=(Nimages,image_height,image_width,3),dtype='float32')
# Y = np.zeros(shape = Nimages) 

# import json
# import os
# import numpy as np
# import PIL.Image
# import cv2
# import matplotlib.pyplot as plt

# # import albumentations as A
    
# # transform = A.Compose([
# #     A.Defocus(radius = 5, alias_blur = 0.2, p = 0.5),
# #     A.ElasticTransform(p =  0.5),
# #     A.VerticalFlip(p=0.5),
# #     A.GridDistortion(p=0.5)
# #     ])



    
# i = 0
# for w in range(np.size(labels, 0)):
#     print('loading image ',w)
#     img_file = labels.iloc[w,0]
#     if labels.iloc[w,1] == '0.0':
#          img = cv2.imread(img_folder_sensitive+'/'+img_file)
# #         img_edgeA = cv2.imread(edge_folder_sensitive_A+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
# #         img_edgeB = cv2.imread(edge_folder_sensitive_B+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
# #        img_fourier = cv2.imread(fourier_folder_sensitive+'/Fourier '+img_file,cv2.IMREAD_GRAYSCALE)
#          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#          Y[i] = 0
#     else:
#         img = cv2.imread(img_folder_resistant+'/'+img_file)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #        img_edgeA = cv2.imread(edge_folder_resistant_A+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
# #        img_edgeB = cv2.imread(edge_folder_resistant_B+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
# #        img_fourier = cv2.imread(fourier_folder_resistant+'/Fourier '+img_file,cv2.IMREAD_GRAYSCALE)
#         Y[i] = 1
#     #normalize pixels
#     img=img/img.max()
# #    img_edgeA=img_edgeA/img_edgeA.max()
# #    img_edgeB=img_edgeB/img_edgeB.max()
#  #   img_fourier=img_fourier/img_fourier.max()
#     X[i,:,:,0:3]=img
# #    X[i,:,:,3]=img_edgeA
# #    X[i,:,:,3]=img_edgeB
# #    X[i,:,:,3]=img_fourier
    
#     i=i+1

# # #%% split into train and test sets - By patient ID

# # #Patient testing IDs: 73 65 100 36 102 97
# # #Patient training IDs: 5 6 50 75 81 92 99 112 119 7 8 28 33 34 43 47 66 85 108 121 140

# # testing_indices = np.array([73, 65, 100, 36, 102,97])
# # training_indices = np.array([5,6,50,75,81,92,99,112,119,7,8,28,33,34,43,47,66,85,108,121,140])

# # ix_tr = []
# # ix_ts = []

# # for i in range(np.size(labels, 0)):
# #     for j in range(6):
# #         if str(testing_indices[j]) in str(wells[i]):
# #             ix_ts.append(i)

         
# # ix_ts = list(set(ix_ts))

# # ix_tr = [x for x in range(np.size(labels, 0)) if x not in ix_ts]


# # # ix = np.arange(len(wells))

# # # #First split into training and validation/test
# # # ix_tr, ix_val_ts = train_test_split(ix,train_size=0.8, random_state=0)

# # # #Then split the validation/test set into validation + test separate
# # # ix_val, ix_ts = train_test_split(ix_val_ts,train_size=0.5, random_state=0)

# # #sanity check, no overlap between train, validation and test sets
# # #assert len(np.intersect1d(ix_tr,ix_val))==0
# # assert len(np.intersect1d(ix_tr,ix_ts))==0
# # #assert len(np.intersect1d(ix_val,ix_ts))==0

# # X_tr = X[ix_tr,:]
# # #X_val = X[ix_val,:]
# # X_ts = X[ix_ts,:]

# # Y_tr = Y[ix_tr,]
# # #Y_val = Y[ix_val]
# # Y_ts = Y[ix_ts]

# # wells_training=wells[ix_tr]
# # wells_testing = wells[ix_ts]

# # for w in range(630):
# #     img = X_tr[w,:,:,:]
# #     print('transforming image',w)
# #     img = img.astype(np.uint8)
# #     transformed_image_1 = transform(image=img)['image']
# #     transformed_image_1 = transformed_image_1[np.newaxis, ... ]
# #     transformed_image_2 = transform(image=img)['image']
# #     transformed_image_2 = transformed_image_2[np.newaxis, ... ]

    
# #     X_tr = np.vstack([X_tr, transformed_image_1])
# #     X_tr = np.vstack([X_tr, transformed_image_2])

       
# #     Y_tr = np.append(Y_tr,Y_tr[w])
# #     Y_tr = np.append(Y_tr,Y_tr[w])




# # # # fnames_tr = wells[ix_tr].tolist()
# # # # fnames_val = wells[ix_val].tolist()
# # # # fnames_ts = wells[ix_ts].tolist()

# # # # fname_split = ['train']*len(fnames_tr)+['validation']*len(fnames_val)+['test']*len(fnames_ts)
# # # # df=pd.DataFrame({'well':fnames_tr+fnames_val+fnames_ts,
# # # #               'split':fname_split})

# # # # #sav
# # # # df.to_csv('training_validation_test_splits.csv',index=False)

# # np.save('X_tr.npy',X_tr)
# # #np.save('X_val.npy',X_val)
# # np.save('X_ts.npy',X_ts)

# # np.save('Y_tr.npy',Y_tr)
# # #np.save('Y_val.npy',Y_val)
# # np.save('Y_ts.npy',Y_ts)

# #%% split into train and test sets - By lesion ID

# testing_indices = np.array(['Chemo_119.1', 'Chemo_34.1', 'Chemo_7.2', 'Chemo_85.6', 'Chemo_92.2' ,'Chemo_5.1' ,'Chemo_28.1', 'Chemo_47.2', 'Chemo_81.6'])#, 'Chemo_119.1', 'Chemo_8.1',  'Chemo_73.2', 'Chemo_100.6', 'Chemo_36', 'Chemo_102.1', 'Chemo_121.3', 'Chemo_81.7'])

# ix_tr = []
# ix_ts = []

# for i in range(np.size(labels, 0)):
#     for j in range(9):
#         if str(testing_indices[j]) in str(wells[i]):
#             ix_ts.append(i)

         
# ix_ts = list(set(ix_ts))

# ix_tr = [x for x in range(np.size(labels, 0)) if x not in ix_ts]
# import random


# #sanity check, no overlap between train, validation and test sets
# assert len(np.intersect1d(ix_tr,ix_ts))==0

# X_tr = X[ix_tr,:]
# X_ts = X[ix_ts,:]

# Y_tr = Y[ix_tr,]
# Y_ts = Y[ix_ts]

# wells_training=wells[ix_tr]
# wells_testing = wells[ix_ts]

# # for w in range(680):
# #     img = X_tr[w,:,:,0:3]
# #     print('transforming image',w)
# #     #img = img.astype(np.uint8)
# #     transformed_image_1 = transform(image=img)['image']
# #     transformed_image_1 = transformed_image_1/transformed_image_1.max()
# #     transformed_image_1 = transformed_image_1[np.newaxis, ... ]
    
# #     # transformed_image_2 = transform(image=img)['image']
# #     # transformed_image_2 = transformed_image_2/transformed_image_2.max()
# #     # transformed_image_2 = transformed_image_2[np.newaxis, ... ]
    
# #     X_tr = np.vstack([X_tr, transformed_image_1])
# #     #X_tr = np.vstack([X_tr, transformed_image_2])

           
# #     Y_tr = np.append(Y_tr,Y_tr[w])
# #     #Y_tr = np.append(Y_tr,Y_tr[w])



# fnames_tr = wells[ix_tr].tolist()
# fnames_ts = wells[ix_ts].tolist()

# fname_split = ['train']*len(fnames_tr)+['test']*len(fnames_ts)
# df=pd.DataFrame({'well':fnames_tr+fnames_ts,
#                'split':fname_split})

# #save training and testing splits
# #df.to_csv('gen/training_test_splits.csv',index=False)

# #comment below to not saving the images on the github
# # np.save('gen/X_tr.npy',X_tr)
# # #np.save('X_val.npy',X_val)
# # np.save('gen/X_ts.npy',X_ts)

# # np.save('gen/Y_tr.npy',Y_tr)
# # #np.save('Y_val.npy',Y_val)
# # np.save('gen/Y_ts.npy',Y_ts)

# #save the training and testing splits to box folder
# df.to_csv('../disc_research_images/gen/training_test_splits_suspicious.csv',index=False)
# #save the training and testing file to box folder

# np.save('../disc_research_images/gen/X_tr_suspicious.npy',X_tr)
# #np.save('X_val.npy',X_val)
# np.save('../disc_research_images/gen/X_ts_suspicious.npy',X_ts)

# np.save('../disc_research_images/gen/Y_tr_suspicious.npy',Y_tr)
# #np.save('Y_val.npy',Y_val)
# np.save('../disc_research_images/gen/Y_ts_suspicious.npy',Y_ts)


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
    for img, label, (patient_lesion_id, replicate) in zip(images, labels, info):
        # Assuming the format of patient_lesion_id is 'patientID.lesionID'
        key = patient_lesion_id

        if key not in grouped_data:
            grouped_data[key] = {'images': [], 'labels': [], 'info': []}
        grouped_data[key]['images'].append(img)
        grouped_data[key]['labels'].append(label)
        grouped_data[key]['info'].append(patient_lesion_id) 
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
train, test = train_test_split(grouped_list, test_size=0.5, random_state=45)


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
all_df.to_csv('../disc_research_images/gen/biopsied/patient_lesion/5050/45/patient_lesion_groups.csv', index=False)

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



np.save('../disc_research_images/gen/biopsied/patient_lesion/5050/45/X_train.npy', X_train)
np.save('../disc_research_images/gen/biopsied/patient_lesion/5050/45/y_train.npy', Y_train)
#np.save('../disc_research_images/gen/suspicious/X_val.npy', X_val)
#np.save('../disc_research_images/gen/suspicious/Y_val.npy', Y_val)
np.save('../disc_research_images/gen/biopsied/patient_lesion/5050/45/X_test.npy', X_test)
np.save('../disc_research_images/gen/biopsied/patient_lesion/5050/45/y_test.npy', Y_test)

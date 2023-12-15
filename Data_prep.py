#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Georgios Georgalis
"""
from PIL import Image
import numpy as np
import pandas as pd

from keras.preprocessing import image
import os

from sklearn.model_selection import train_test_split
import cv2
import csv


#%% Data_prep.py loads the image files and exports the appropriate data structures for training
img_folder_sensitive = './../DISC_StudyGrp_DLChemoResistance/Data_Chemotherapy Resistance/Biopsied Lesions/Sensitive' #Folder that contains the sensitive biopsied RGB images
img_folder_resistant = './../DISC_StudyGrp_DLChemoResistance/Data_Chemotherapy Resistance/Biopsied Lesions/Resistant' #Folder that contains the resistant biopsied RGB images
#edge_folder_sensitive_A ='./../../Images_Biopsied/Sensitive_Canny_0.4'
#edge_folder_resistant_A ='./../../Images_Biopsied/Resistant_Canny_0.4'
#Path to .csv with the labels (0 = sensitive, 1 = resistant)

#Image dimensions
image_height = 80
image_width = 80


# load filenames
img_filenames_sensitive = np.array(sorted(os.listdir(img_folder_sensitive)))#sort
img_filenames_resistant = np.array(sorted(os.listdir(img_folder_resistant)))#sort

d1 = np.transpose(np.vstack([img_filenames_sensitive, np.zeros(np.size(img_filenames_sensitive))]))
d2 = np.transpose(np.vstack([img_filenames_resistant, np.ones(np.size(img_filenames_resistant))]))
d = np.vstack([d1, d2])

labels = pd.DataFrame(data = d, columns = ['image_name', 'label'])


wells = labels['image_name']
Nimages = np.size(wells) #Number of images

#Empty arrays for RGB images (X)
X = np.zeros(shape=(Nimages,image_height,image_width,3),dtype='float32')
Y = np.zeros(shape = Nimages) 

import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt

# import albumentations as A
    
# transform = A.Compose([
#     A.Defocus(radius = 5, alias_blur = 0.2, p = 0.5),
#     A.ElasticTransform(p =  0.5),
#     A.VerticalFlip(p=0.5),
#     A.GridDistortion(p=0.5)
#     ])



    
i = 0
for w in range(np.size(labels, 0)):
    print('loading image ',w)
    img_file = labels.iloc[w,0]
    if labels.iloc[w,1] == '0.0':
         img = cv2.imread(img_folder_sensitive+'/'+img_file)
#         img_edgeA = cv2.imread(edge_folder_sensitive_A+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
#         img_edgeB = cv2.imread(edge_folder_sensitive_B+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
#        img_fourier = cv2.imread(fourier_folder_sensitive+'/Fourier '+img_file,cv2.IMREAD_GRAYSCALE)
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         Y[i] = 0
    else:
        img = cv2.imread(img_folder_resistant+'/'+img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        img_edgeA = cv2.imread(edge_folder_resistant_A+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
#        img_edgeB = cv2.imread(edge_folder_resistant_B+'/Canny '+img_file,cv2.IMREAD_GRAYSCALE)
#        img_fourier = cv2.imread(fourier_folder_resistant+'/Fourier '+img_file,cv2.IMREAD_GRAYSCALE)
        Y[i] = 1
    #normalize pixels
    img=img/img.max()
#    img_edgeA=img_edgeA/img_edgeA.max()
#    img_edgeB=img_edgeB/img_edgeB.max()
 #   img_fourier=img_fourier/img_fourier.max()
    X[i,:,:,0:3]=img
#    X[i,:,:,3]=img_edgeA
#    X[i,:,:,3]=img_edgeB
#    X[i,:,:,3]=img_fourier
    
    i=i+1

# #%% split into train and test sets - By patient ID

# #Patient testing IDs: 73 65 100 36 102 97
# #Patient training IDs: 5 6 50 75 81 92 99 112 119 7 8 28 33 34 43 47 66 85 108 121 140

# testing_indices = np.array([73, 65, 100, 36, 102,97])
# training_indices = np.array([5,6,50,75,81,92,99,112,119,7,8,28,33,34,43,47,66,85,108,121,140])

# ix_tr = []
# ix_ts = []

# for i in range(np.size(labels, 0)):
#     for j in range(6):
#         if str(testing_indices[j]) in str(wells[i]):
#             ix_ts.append(i)

         
# ix_ts = list(set(ix_ts))

# ix_tr = [x for x in range(np.size(labels, 0)) if x not in ix_ts]


# # ix = np.arange(len(wells))

# # #First split into training and validation/test
# # ix_tr, ix_val_ts = train_test_split(ix,train_size=0.8, random_state=0)

# # #Then split the validation/test set into validation + test separate
# # ix_val, ix_ts = train_test_split(ix_val_ts,train_size=0.5, random_state=0)

# #sanity check, no overlap between train, validation and test sets
# #assert len(np.intersect1d(ix_tr,ix_val))==0
# assert len(np.intersect1d(ix_tr,ix_ts))==0
# #assert len(np.intersect1d(ix_val,ix_ts))==0

# X_tr = X[ix_tr,:]
# #X_val = X[ix_val,:]
# X_ts = X[ix_ts,:]

# Y_tr = Y[ix_tr,]
# #Y_val = Y[ix_val]
# Y_ts = Y[ix_ts]

# wells_training=wells[ix_tr]
# wells_testing = wells[ix_ts]

# for w in range(630):
#     img = X_tr[w,:,:,:]
#     print('transforming image',w)
#     img = img.astype(np.uint8)
#     transformed_image_1 = transform(image=img)['image']
#     transformed_image_1 = transformed_image_1[np.newaxis, ... ]
#     transformed_image_2 = transform(image=img)['image']
#     transformed_image_2 = transformed_image_2[np.newaxis, ... ]

    
#     X_tr = np.vstack([X_tr, transformed_image_1])
#     X_tr = np.vstack([X_tr, transformed_image_2])

       
#     Y_tr = np.append(Y_tr,Y_tr[w])
#     Y_tr = np.append(Y_tr,Y_tr[w])




# # # fnames_tr = wells[ix_tr].tolist()
# # # fnames_val = wells[ix_val].tolist()
# # # fnames_ts = wells[ix_ts].tolist()

# # # fname_split = ['train']*len(fnames_tr)+['validation']*len(fnames_val)+['test']*len(fnames_ts)
# # # df=pd.DataFrame({'well':fnames_tr+fnames_val+fnames_ts,
# # #               'split':fname_split})

# # # #sav
# # # df.to_csv('training_validation_test_splits.csv',index=False)

# np.save('X_tr.npy',X_tr)
# #np.save('X_val.npy',X_val)
# np.save('X_ts.npy',X_ts)

# np.save('Y_tr.npy',Y_tr)
# #np.save('Y_val.npy',Y_val)
# np.save('Y_ts.npy',Y_ts)

#%% split into train and test sets - By lesion ID

testing_indices = np.array(['Chemo_119.1', 'Chemo_34.1', 'Chemo_7.2', 'Chemo_85.6', 'Chemo_92.2' ,'Chemo_5.1' ,'Chemo_28.1', 'Chemo_47.2', 'Chemo_81.6'])#, 'Chemo_119.1', 'Chemo_8.1',  'Chemo_73.2', 'Chemo_100.6', 'Chemo_36', 'Chemo_102.1', 'Chemo_121.3', 'Chemo_81.7'])

ix_tr = []
ix_ts = []

for i in range(np.size(labels, 0)):
    for j in range(9):
        if str(testing_indices[j]) in str(wells[i]):
            ix_ts.append(i)

         
ix_ts = list(set(ix_ts))

ix_tr = [x for x in range(np.size(labels, 0)) if x not in ix_ts]
import random


#sanity check, no overlap between train, validation and test sets
assert len(np.intersect1d(ix_tr,ix_ts))==0

X_tr = X[ix_tr,:]
X_ts = X[ix_ts,:]

Y_tr = Y[ix_tr,]
Y_ts = Y[ix_ts]

wells_training=wells[ix_tr]
wells_testing = wells[ix_ts]

# for w in range(680):
#     img = X_tr[w,:,:,0:3]
#     print('transforming image',w)
#     #img = img.astype(np.uint8)
#     transformed_image_1 = transform(image=img)['image']
#     transformed_image_1 = transformed_image_1/transformed_image_1.max()
#     transformed_image_1 = transformed_image_1[np.newaxis, ... ]
    
#     # transformed_image_2 = transform(image=img)['image']
#     # transformed_image_2 = transformed_image_2/transformed_image_2.max()
#     # transformed_image_2 = transformed_image_2[np.newaxis, ... ]
    
#     X_tr = np.vstack([X_tr, transformed_image_1])
#     #X_tr = np.vstack([X_tr, transformed_image_2])

           
#     Y_tr = np.append(Y_tr,Y_tr[w])
#     #Y_tr = np.append(Y_tr,Y_tr[w])



fnames_tr = wells[ix_tr].tolist()
fnames_ts = wells[ix_ts].tolist()

fname_split = ['train']*len(fnames_tr)+['test']*len(fnames_ts)
df=pd.DataFrame({'well':fnames_tr+fnames_ts,
               'split':fname_split})

#save training and testing splits
#df.to_csv('gen/training_test_splits.csv',index=False)

#comment below to not saving the images on the github
# np.save('gen/X_tr.npy',X_tr)
# #np.save('X_val.npy',X_val)
# np.save('gen/X_ts.npy',X_ts)

# np.save('gen/Y_tr.npy',Y_tr)
# #np.save('Y_val.npy',Y_val)
# np.save('gen/Y_ts.npy',Y_ts)

#save the training and testing splits to box folder
df.to_csv('../disc_research_images/gen/training_test_splits.csv',index=False)
#save the training and testing file to box folder

np.save('../disc_research_images/gen/X_tr.npy',X_tr)
#np.save('X_val.npy',X_val)
np.save('../disc_research_images/gen/X_ts.npy',X_ts)

np.save('../disc_research_images/gen/Y_tr.npy',Y_tr)
#np.save('Y_val.npy',Y_val)
np.save('../disc_research_images/gen/Y_ts.npy',Y_ts)
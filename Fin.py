from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import backend as k
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd


import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np

import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
os.listdir('E:\spoofd\\validation')
['client_lebel.csv',
 'client_lebel.xlsx',
 'imposter_lebel.csv',
 'imposter_lebel.xlsx']
data0 = pd.read_csv('E:/spoofd/validation//imposter_lebel.csv')
data1 = pd.read_csv('E:/spoofd/validation//client_lebel.csv')
data1['lebel'].value_counts()
1    150
Name: lebel, dtype: int64
data0.head()
id	lebel
0	0001_00_00_01_0	0
1	0001_00_00_01_5	0
2	0001_00_00_01_13	0
3	0001_00_00_01_18	0
4	0001_00_00_01_22	0
data1.head()
id	lebel
0	0001_00_00_01_0	1
1	0001_00_00_01_2	1
2	0001_00_00_01_6	1
3	0001_00_00_01_12	1
4	0001_00_00_01_16	1
y = data1['lebel']

df_train1, df_val1 = train_test_split(data1, test_size=0.30, random_state=101, stratify=y)
df_train1['lebel'].value_counts()
1    105
Name: lebel, dtype: int64
df_val1['lebel'].value_counts()
1    45
Name: lebel, dtype: int64
y = data0['lebel']

df_train0, df_val0 = train_test_split(data0, test_size=0.30, random_state=101, stratify=y)
df_train0['lebel'].value_counts()
0    105
Name: lebel, dtype: int64
df_val0['lebel'].value_counts()
0    45
Name: lebel, dtype: int64
df_val = pd.concat([df_val0, df_val1], axis=0).reset_index(drop=True)
df_train = pd.concat([df_train0, df_train1], axis=0).reset_index(drop=True)
 
# Create a new directory
base_dir1 = 'base_dir1'
os.mkdir(base_dir1)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 2 folders inside 'base_dir':

# train_dir
    # a_no_tumor_tissue
    # b_has_tumor_tissue

# val_dir
    # a_no_tumor_tissue
    # b_has_tumor_tissue



# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir1, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir1, 'val_dir')
os.mkdir(val_dir)



# [CREATE FOLDERS INSIDE THE TRAIN AND VALIDATION FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train_dir
client_image = os.path.join(train_dir, 'client_image')
os.mkdir(client_image)
imposed_image = os.path.join(train_dir, 'imposed_image')
os.mkdir(imposed_image)


# create new folders inside val_dir
client_image = os.path.join(val_dir, 'client_image')
os.mkdir(client_image)
imposed_image = os.path.join(val_dir, 'imposed_image')
os.mkdir(imposed_image)
os.listdir('base_dir1/train_dir')
['client_image', 'imposed_image']
os.listdir('base_dir1/val_dir')
['client_image', 'imposed_image']
data0.set_index('id', inplace=True)
data1.set_index('id', inplace=True)
# Get a list of train and val images
train_list = list(df_train1['id'])
val_list = list(df_val1['id'])



# Transfer the train images

for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.jpg'
    # get the label for a certain image
    target = data1.loc[image,'lebel']
    
    # these must match the folder names
    if target == 1:
        label = 'client_image'
    if target == 0:
        label = 'imposed_image'
    
    # source path to image
    src = os.path.join('E:/spoofd/train//real', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    
    shutil.copyfile(src, dst)
    
    # Transfer the val images

    
    # Transfer the val images
val_list = list(df_val1['id'])

for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.jpg'
    # get the label for a certain image
    target = data1.loc[image,'lebel']
    
    # these must match the folder names
    if target == 1:
        label = 'client_image'
    if target == 0:
        label = 'imposed_image'
    
    # source path to image
    src = os.path.join('E:/spoofd/train//real',fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    
    shutil.copyfile(src, dst)
    
    # Transfer the val images

    
    # Transfer the val images
# Get a list of train and val images
train_list0 = list(df_train0['id'])
val_list0 = list(df_val0['id'])



# Transfer the train images

for image in train_list0:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.jpg'
    # get the label for a certain image
    target = data0.loc[image,'lebel']
    
    # these must match the folder names
    if target == 1:
        label = 'client_image'
    if target == 0:
        label = 'imposed_image'
    
    # source path to image
    src = os.path.join('E:/spoofd/train//imposed', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    
    shutil.copyfile(src, dst)
    
    # Transfer the val images

    
    # Transfer the val images


    
for image in val_list0:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.jpg'
    # get the label for a certain image
    target = data0.loc[image,'lebel']
    
    # these must match the folder names
    if target == 1:
        label = 'client_image'
    if target == 0:
        label = 'imposed_image'
    
    # source path to image
    src = os.path.join('E:/spoofd/train//imposed',fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    
    shutil.copyfile(src, dst)
print(len(os.listdir('base_dir1/train_dir/client_image')))
print(len(os.listdir('base_dir1/train_dir/imposed_image')))
105
105
print(len(os.listdir('base_dir1/val_dir/client_image')))
print(len(os.listdir('base_dir1/val_dir/imposed_image')))
45
45
train_path = 'base_dir1/train_dir'
valid_path = 'base_dir1/val_dir'
test_path = 'E:\\spoofd\\test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
IMAGE_SIZE,IMAGE_SIZE=203,203


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')


test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                       
                                        shuffle=False)
Found 210 images belonging to 2 classes.
Found 90 images belonging to 2 classes.
Found 90 images belonging to 2 classes.
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 64
second_filters = 128
third_filters = 256

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))




model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_16 (Conv2D)           (None, 94, 94, 64)        1792      
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 92, 92, 64)        36928     
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 90, 90, 64)        36928     
_________________________________________________________________
max_pooling2d_16 (MaxPooling (None, 45, 45, 64)        0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 45, 45, 64)        0         
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 43, 43, 128)       73856     
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 41, 41, 128)       147584    
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 39, 39, 128)       147584    
_________________________________________________________________
max_pooling2d_17 (MaxPooling (None, 19, 19, 128)       0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 19, 19, 128)       0         
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 17, 17, 256)       295168    
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 15, 15, 256)       590080    
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 13, 13, 256)       590080    
_________________________________________________________________
max_pooling2d_18 (MaxPooling (None, 6, 6, 256)         0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 6, 6, 256)         0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_11 (Dense)             (None, 256)               2359552   
_________________________________________________________________
dropout_9 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_12 (Dense)             (None, 2)                 514       
=================================================================
Total params: 4,280,066
Trainable params: 4,280,066
Non-trainable params: 0
_________________________________________________________________
model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])
# Get the labels that are associated with each index
print(val_gen.class_indices)
{'client_image': 0, 'imposed_image': 1}
filepath = 'modela.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.01)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=20, verbose=1,
                  callbacks=callbacks_list)
Epoch 1/20
20/21 [===========================>..] - ETA: 0s - loss: 0.4624 - acc: 0.8200Epoch 1/20
 9/21 [===========>..................] - ETA: 3s - loss: 0.2522 - acc: 0.9222
Epoch 00001: val_acc improved from -inf to 0.92222, saving model to modela.h5
21/21 [==============================] - 20s 973ms/step - loss: 0.4692 - acc: 0.8143 - val_loss: 0.2522 - val_acc: 0.9222
Epoch 2/20
20/21 [===========================>..] - ETA: 0s - loss: 0.2664 - acc: 0.8950Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.3020 - acc: 0.8556
Epoch 00002: val_acc did not improve from 0.92222
21/21 [==============================] - 16s 771ms/step - loss: 0.2648 - acc: 0.8952 - val_loss: 0.3020 - val_acc: 0.8556
Epoch 3/20
20/21 [===========================>..] - ETA: 0s - loss: 0.2358 - acc: 0.9200Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.1564 - acc: 0.9889
Epoch 00003: val_acc improved from 0.92222 to 0.98889, saving model to modela.h5
21/21 [==============================] - 16s 762ms/step - loss: 0.2357 - acc: 0.9190 - val_loss: 0.1564 - val_acc: 0.9889
Epoch 4/20
20/21 [===========================>..] - ETA: 0s - loss: 0.1419 - acc: 0.9700Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0879 - acc: 1.0000
Epoch 00004: val_acc improved from 0.98889 to 1.00000, saving model to modela.h5
21/21 [==============================] - 16s 780ms/step - loss: 0.1385 - acc: 0.9714 - val_loss: 0.0879 - val_acc: 1.0000
Epoch 5/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0762 - acc: 0.9850Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0407 - acc: 1.0000
Epoch 00005: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 764ms/step - loss: 0.0739 - acc: 0.9857 - val_loss: 0.0407 - val_acc: 1.0000
Epoch 6/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0456 - acc: 0.9950Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0894 - acc: 0.9667
Epoch 00006: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 761ms/step - loss: 0.0460 - acc: 0.9952 - val_loss: 0.0894 - val_acc: 0.9667
Epoch 7/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0810 - acc: 0.9650Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0852 - acc: 0.9778
Epoch 00007: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 771ms/step - loss: 0.0795 - acc: 0.9667 - val_loss: 0.0852 - val_acc: 0.9778
Epoch 8/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0450 - acc: 0.9950Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0366 - acc: 0.9889
Epoch 00008: val_acc did not improve from 1.00000
21/21 [==============================] - 17s 796ms/step - loss: 0.0443 - acc: 0.9952 - val_loss: 0.0366 - val_acc: 0.9889
Epoch 9/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0280 - acc: 0.9950Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0107 - acc: 1.0000
Epoch 00009: val_acc did not improve from 1.00000
21/21 [==============================] - 17s 790ms/step - loss: 0.0310 - acc: 0.9905 - val_loss: 0.0107 - val_acc: 1.0000
Epoch 10/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0143 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0075 - acc: 1.0000
Epoch 00010: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 774ms/step - loss: 0.0139 - acc: 1.0000 - val_loss: 0.0075 - val_acc: 1.0000
Epoch 11/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0249 - acc: 0.9900Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0071 - acc: 1.0000
Epoch 00011: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 769ms/step - loss: 0.0249 - acc: 0.9905 - val_loss: 0.0071 - val_acc: 1.0000
Epoch 12/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0224 - acc: 0.9900Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0164 - acc: 1.0000
Epoch 00012: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 767ms/step - loss: 0.0227 - acc: 0.9905 - val_loss: 0.0164 - val_acc: 1.0000
Epoch 13/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0151 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0044 - acc: 1.0000
Epoch 00013: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 764ms/step - loss: 0.0196 - acc: 0.9952 - val_loss: 0.0044 - val_acc: 1.0000
Epoch 14/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0276 - acc: 0.9950Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0051 - acc: 1.0000
Epoch 00014: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 778ms/step - loss: 0.0270 - acc: 0.9952 - val_loss: 0.0051 - val_acc: 1.0000
Epoch 15/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0086 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0047 - acc: 1.0000
Epoch 00015: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 775ms/step - loss: 0.0083 - acc: 1.0000 - val_loss: 0.0047 - val_acc: 1.0000
Epoch 16/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0052 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0036 - acc: 1.0000
Epoch 00016: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 768ms/step - loss: 0.0055 - acc: 1.0000 - val_loss: 0.0036 - val_acc: 1.0000
Epoch 17/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0213 - acc: 0.9900Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0033 - acc: 1.0000
Epoch 00017: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 774ms/step - loss: 0.0204 - acc: 0.9905 - val_loss: 0.0033 - val_acc: 1.0000
Epoch 18/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0100 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0057 - acc: 1.0000
Epoch 00018: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 785ms/step - loss: 0.0096 - acc: 1.0000 - val_loss: 0.0057 - val_acc: 1.0000
Epoch 19/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0055 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0017 - acc: 1.0000
Epoch 00019: val_acc did not improve from 1.00000
21/21 [==============================] - 17s 789ms/step - loss: 0.0053 - acc: 1.0000 - val_loss: 0.0017 - val_acc: 1.0000
Epoch 20/20
20/21 [===========================>..] - ETA: 0s - loss: 0.0034 - acc: 1.0000Epoch 1/20
 9/21 [===========>..................] - ETA: 2s - loss: 0.0013 - acc: 1.0000    
Epoch 00020: val_acc did not improve from 1.00000
21/21 [==============================] - 16s 768ms/step - loss: 0.0035 - acc: 1.0000 - val_loss: 0.0013 - val_acc: 1.0000
model.metrics_names
['loss', 'acc']
model.load_weights('modela.h5')

val_loss, val_acc = \
model.evaluate_generator(test_gen, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)
val_loss: 0.08789071377703092
val_acc: 1.0
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'red', label='Training loss',)
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
<Figure size 432x288 with 0 Axes>


<Figure size 432x288 with 0 Axes>
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
90/90 [==============================] - 3s 35ms/step
predictions.shape
(90, 2)
test_gen.class_indices
{'client_image': 0, 'imposed_image': 1}
df_preds = pd.DataFrame(predictions, columns=['client_image', 'imposed_image'])

df_preds.head()
client_image	imposed_image
0	0.997290	0.002710
1	0.998282	0.001718
2	0.998105	0.001895
3	0.998100	0.001900
4	0.997159	0.002841
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['imposed_image']
from sklearn.metrics import roc_auc_score

roc_auc_score(y_true, y_pred)
1.0
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
test_labels = test_gen.classes
test_labels.shape
(90,)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
test_gen.class_indices
{'client_image': 0, 'imposed_image': 1}
cm_plot_labels = ['client_image', 'imposed_image']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
Confusion matrix, without normalization
[[45  0]
 [ 0 45]]

import tkinter as tk
from tkinter import *

def Action():
   
    img_pred = image.load_img(e1.get(), target_size=(203,203))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis =0)
    
    value = model.predict(img_pred)
    print (value)
    if value [0][0] ==1:
        prediction='   Real Image    '
    else:
        prediction='Imposed Image'
    print(prediction)
    label = Label(master, text= prediction,font=('Arial Black',10))
    label.grid(row=2,column=9)

master = tk.Tk()
master.title("Identifier")
master.configure(background='violet')
master.geometry('550x150')
e1 = tk.Entry(master)
e1.grid(row=2, column=7,padx=10)

tk.Label(master, text="Enter Image path").grid(row=2,padx=5)

tk.Button(master, text = 'Enter', command = Action,bg='red').grid(row = 2, column = 8, pady = 10,padx=20)


                                                        
                                                        
master.mainloop()
[[0. 1.]]
Imposed Image
[[1. 0.]]
   Real Image    
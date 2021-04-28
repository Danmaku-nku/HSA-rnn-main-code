# -*- coding: utf-8 -*-

train_video_number=1
test_video_number=1

import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

"""Pre-importing some required libraries (others will be imported as and when required further)"""

import cv2  # for capturing videos
import matplotlib.pyplot as plt  # for plotting the images
import pandas as pd
from keras.preprocessing import image  # for preprocessing the images
import numpy as np  # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize  # for resizing images
import os

"""DATA PREPROCESSING (X:total_no_of_frames_of_50vids x 240 x240 and Y: total_no_of_frames,1)

Preprocessing Y/Label Data
"""

print('DATA PREPROCESSING BEGIN')

df = pd.read_csv("../ydata-tvsum50-v1_1/ydata-tvsum50-data/test/ydata-tvsum50-anno.tsv", header=None,
                 error_bad_lines=False, sep="\t")
df_edit = pd.DataFrame()

video_number=test_video_number+train_video_number

for i in range(video_number):
    df_edit = df_edit.append(df.loc[i * 20 + 1], ignore_index=True)

df_edit = df_edit.sort_values(0).iloc[:]
df_edit = df_edit.reset_index(drop=True)

new_scores = []
for i in range(df_edit.shape[0]):
    temp = df_edit[2][i].split(',')
    for j in range(len(temp)):
        new_scores.append(int(temp[j]))
    df_edit[2][i] = new_scores
    new_scores = []

df_edit.head()



"""PREPROCESSING IMAGE DATA/X """

# DIR = "/content/drive/My Drive/HSA-RNN/"
# DIR = "HSA-RNN/"
DIR = "../"
CATEGORIES = ["ydata-tvsum50-v1_1/ydata-tvsum50-video/test"]

column_names = ["Video_Name", "Frame_count", "Timestamp"]
Frame_data = pd.DataFrame(columns=column_names)

dataset = []
vidnum = 1
train_frames=0
for category in CATEGORIES:
    path = os.path.join(DIR, category)

    for vid in sorted(os.listdir(path)):
        try:
            print('video_num:%d  video_name:%s' % (vidnum , vid))
            cap = cv2.VideoCapture(os.path.join(path, vid))
            count = 0
            success = 1
            # frameRate = cap.get(5)#get frame rate
            while (success):
                # frameId = cap.get(1)#current frame number
                success, frame = cap.read()
                if (success != True):
                    break
                filename = "frame%d.jpg" % count;
                count += 1
                cv2.imwrite(filename, frame)
                img_array = cv2.imread(filename, cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (224, 224))
                dataset.append(new_array)
                Frame_data = Frame_data.append({"Video_Name": vid, "Frame_count": len(dataset),
                                                "Timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000}, ignore_index=True)
                os.remove(filename)
                if vidnum <= train_video_number:
                    train_frames += 1
            vidnum += 1
            print("dataset_len:%d"%len(dataset))
            cap.release()
        except Exception as e:
            print(e)
            pass

sum_frames = len(dataset)

#print(sum_frames)

"""Padding data (NOT USING)"""

# from keras.preprocessing.sequence import pad_sequences
# padded_frames = pad_sequences(dataset)

# len(padded_frames[0])

# len(padded_frames[4])

"""X/Image data DATA TO NUMPY ARRAY

"""

padded_frames = dataset

padded_frames = np.array(padded_frames)

"""Divide test and train"""

padded_frames_train = padded_frames[0:train_frames]

padded_frames_test = padded_frames[train_frames:]

"""Preprocessing Y/Labels data"""

Y_padded = df_edit[2]

Y_long = []
for j in range(video_number):
    for k in Y_padded[j]:
        Y_long.append(k)

#print(len(Y_long))
Y_long = np.array(Y_long)
#print(Y_long.shape)


"""One Hot Encoding the Y data """

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
Y_long = Y_long.reshape(len(Y_long), 1)
Y_long = onehot_encoder.fit_transform(Y_long)
#print(Y_long.shape)

Y_padded_train = Y_long[0:train_frames]
#print(Y_padded_train.shape)

Y_padded_test = Y_long[train_frames:]
#print(Y_padded_test.shape)

print('DATA PREPROCESSING SUCCESS')

"""Extracting image features from VGG pretrained Model"""

print('Extract Begin')

from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Flatten, Input, GRU, Attention, Concatenate, \
    TimeDistributed
from tensorflow.keras.optimizers import RMSprop

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_base.summary()

features_extracted = conv_base.predict(padded_frames_train)


X_train_features_extracted = features_extracted.reshape(train_frames, 7 * 7, 512)


# conv_base.save_weights("/content/drive/My Drive/HSA-RNN/vgg_weights_imgnet.h5")



"""#Model"""

print('Train Begin')

encoder_inputs = Input(shape=(49, 512))

encoder_BidirectionalLSTM = Bidirectional(LSTM(64, return_sequences=True))
encoder_out = encoder_BidirectionalLSTM(encoder_inputs)

decoder_BidirectionalLSTM = Bidirectional(LSTM(64, return_sequences=True))
decoder_out = decoder_BidirectionalLSTM(encoder_out)

attn_layer = Attention(use_scale=True)
attn_out = attn_layer([encoder_out, decoder_out])

dense = Dense(128, activation='relu')
decoder_pred = dense(attn_out)

d1 = LSTM(64, dropout=0.5)(decoder_pred)

d2 = Dense(5, activation="softmax")(d1)

from tensorflow.keras.models import Model

model = Model(inputs=encoder_inputs, outputs=d2)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

'''
#FOR MORE EPOCHS USE THIS:
initial_learning_rate = 0.00001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.9,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9,epsilon=1e-05,),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''

model.summary()

model.fit(X_train_features_extracted, Y_padded_train, batch_size=32, epochs=5)  # Put epoch value accordingly

print('Train End')

'''# save model'''

print("Saving Model Begin")
mp = "./model.h5"
model.save(mp)

"""Testing"""

print('Test Set Process Begin')

features_extracted_test = conv_base.predict(padded_frames_test)

X_test_features_extracted = features_extracted_test.reshape(sum_frames-train_frames, 7 * 7, 512)

Y_pred = model.predict(X_test_features_extracted)

# Y_pred.shape should match Y_padded_test.shape

#print(Y_pred)

Y_edited = np.zeros_like(Y_pred)
Y_edited[np.arange(len(Y_pred)), Y_pred.argmax(1)] = 1

#print(Y_edited)

pred_last=Y_edited.argmax(1)+1

#print('pred_last:')
#print(pred_last)
'''           
# f-measure check
from sklearn.metrics import f1_score

f1_score(Y_padded_test, Y_edited, average='micro')
'''

"""Evaluation Code"""

print('Evaluation Begin')
evaluation = model.evaluate(X_test_features_extracted, Y_padded_test, batch_size=128)


resulting_frames_actual = []
resulting_frames_predicted = []

for i in range(sum_frames-train_frames):
    if (Y_padded_test[i][3] == 1 or Y_padded_test[i][4] == 1):
        resulting_frames_actual.append(i)

column_names = ["Video_Name", "Frame_count", "Timestamp"]
Frame_video_actual = pd.DataFrame(columns=column_names)

for i in resulting_frames_actual:
    Frame_video_actual = Frame_video_actual.append(Frame_data.loc[train_frames + i])

len(Frame_video_actual)

len(resulting_frames_actual)

for i in range(sum_frames-train_frames):
     if (Y_edited[i][3] == 1 or Y_edited[i][4] == 1):
        resulting_frames_predicted.append(i)

column_names = ["Video_Name", "Frame_count", "Timestamp"]
Frame_video_pred = pd.DataFrame(columns=column_names)

for i in resulting_frames_predicted:
    Frame_video_pred = Frame_video_pred.append(Frame_data.loc[train_frames + i])

print('Frame_video_pred.head:')
print(Frame_video_pred.head())

Frame_video_pred.to_csv("./video_pred.csv",index=False)

len(resulting_frames_predicted)

"""Confusion Matrix"""

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_padded_test.argmax(1), Y_edited.argmax(1))



import seaborn as sns

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')

from sklearn.metrics import classification_report

# show a nicely formatted classification report
print(classification_report(Y_padded_test, Y_edited))

print('End Successfully')

'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import DepthwiseConv2D, BatchNormalization, Activation, Input, MaxPool2D

# VGG with Mobilenet
model = Sequential()

####################################VGG16 original######################################################################
# model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
########################################################################################################

##############################VGG using Mobilenet##########################################################
####################### 1st Conv ############################################################################
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
######################## 2nd Conv (Seperable) ##########################################################################
model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), use_bias=False, padding='same', name='Dept_1a'))
model.add(BatchNormalization(name='BN_1.1a'))
model.add(Activation('relu', name='Act_1.1a'))
model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same', name='Point_1a'))
model.add(BatchNormalization(name='BN_1.2a'))
model.add(Activation('relu', name='Act_1.2a'))
##################################### 3nrd Conv #########################################################
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='Max_Pool_1'))
model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), use_bias=False, padding='same', name='Dept_2a'))
model.add(BatchNormalization(name='BN_2.1a'))
model.add(Activation('relu', name='Act_2.1a'))
model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same', name='Point_2a'))
model.add(BatchNormalization(name='BN_2.2a'))
model.add(Activation('relu', name='Act_2.2a'))
##################################### 4th Conv #########################################################
model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), use_bias=False, padding='same', name='Dept_2b'))
model.add(BatchNormalization(name='BN_2.1b'))
model.add(Activation('relu', name='Act_2.1b'))
model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same', name='Point_2b'))
model.add(BatchNormalization(name='BN_2.2b'))
model.add(Activation('relu', name='Act_2.2b'))
##################################### 5th Conv #########################################################
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='Max_Pool_2'))

import keras

model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
'''
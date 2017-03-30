import os.path
import pandas as pd
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc

labelnum = 4
input_width = 224
input_height = 224

flat_width=32
flat_height=32
x=[]

def image_to_feature_vector(image, size=(flat_width, flat_height)):
	return cv2.resize(image, size).flatten()
def loadImage (inFileName, outType ) :
    img = Image.open( inFileName )
    img.load()
    data = np.asarray( img, dtype="float32" )
    if outType == "anArray":
        return data
    if outType == "aList":
        return list(data)
train = pd.read_csv('img.csv').values
trainY=[]
for i in range(0,len(train)):
    imgur = train[i][0]
    imag=cv2.imread(imgur)
    imag=cv2.resize(imag,(input_width,input_height))
    img = img_to_array(imag)
    f=image_to_feature_vector(img)
    x.append(f)
    if (i==0):
        trainY=np_utils.to_categorical(train[i][1],labelnum)
    else:
        trainY=np.row_stack((trainY,np_utils.to_categorical(train[i][1],labelnum)))
    
X_test =np.array(x)/ 255
print(X_test.shape)
print(X_test[0].shape)
model = Sequential()
model.add(Dense(768, activation="relu", kernel_initializer="uniform", input_dim=3072))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(labelnum))
model.add(Activation("softmax"))
sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

model_file = 'csvload.h5'
if os.path.isfile(model_file):
    model.load_weights(model_file)

    try:
        model.fit(X_test, trainY, batch_size=256,epochs=50, verbose=1)
    except KeyboardInterrupt:
        pass
json_string = model.to_json()
filename = "csvload"
open(filename+'.json', 'w').write(json_string)
model.save_weights(filename+'.h5', overwrite=True)
model.save_weights(model_file)

gc.collect()

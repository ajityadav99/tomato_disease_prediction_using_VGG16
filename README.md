# tomato_disease_prediction_using_VGG16

After Training using the VGG16 Architecture we were able to achieve 91.38% accuracy in properly predicting the disease related to tomatoes.

![image](https://user-images.githubusercontent.com/121440744/211755457-1051fe2a-8ec1-4a3d-a5b6-50d8d5ed275c.png)

VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.

![image](https://user-images.githubusercontent.com/121440744/211755599-d29aa222-13bd-4983-ba4d-479fddfac3e0.png)


Importing library

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

As we are using VGG16 architecture, it expects the size of 224 by 224. We will set image size.

image_size = [224, 224]
vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top =  False

The first argument is the shape of input image plus 3(as image is colured[RBG], for black_and_white add 1). The second one is the weights eqaul to imagenet. And, as we know it gives 1000 outputs. Third one excludes the top layer.

for layer in vgg.layers:
    layer.trainable = False
    
Some of the layers of VGG16 are already trained. To train them again is not a good practice. Thereby making it False

from glob import glob
folders = glob('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/*')
folders

Flattening the output layer

x = Flatten()(vgg.output
prediction = Dense(len(folders), activation = 'softmax')(x)
model = Model(inputs = vgg.input, outputs = prediction)
model.summary()

Compiling the model

model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = 'adam', 
    metrics = ['accuracy']
)

Generating more images

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_gen = ImageDataGenerator(rescale = 1./255)
train_set = train_data_gen.flow_from_directory('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/', target_size = (224,224), batch_size = 16, class_mode = 'categorical')

Found 18345 images belonging to 10 classes.

test_set = test_data_gen.flow_from_directory('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/', target_size = (224,224), batch_size = 16, class_mode = 'categorical')
Found 4585 images belonging to 10 classes.
Plotting few images

import matplotlib.pyplot as plt
plt.imshow(plt.imread("../input/tomato/New Plant Diseases Dataset(Augmented)/train/Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp 3110.JPG"))
plt.title("Bacterial Spot")
Text(0.5, 1.0, 'Bacterial Spot')
![image](https://user-images.githubusercontent.com/121440744/211757459-0c927890-1806-4867-979a-6b1a7bd42196.png)

plt.imshow(plt.imread("../input/tomato/New Plant Diseases Dataset(Augmented)/train/Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"))
plt.title("Early Blight")
Text(0.5, 1.0, 'Early Blight')
![image](https://user-images.githubusercontent.com/121440744/211757542-fdc4a339-3ab8-412f-a416-764c2e218790.png)

plt.imshow(plt.imread("../input/tomato/New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight/0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B 4946.JPG"))
plt.title("Late Blight")
Text(0.5, 1.0, 'Late Blight')
![image](https://user-images.githubusercontent.com/121440744/211757609-484ab76e-874f-4065-ba8f-ec0c64469f4a.png)

Fitting the model

mod = model.fit(
  train_set,
  validation_data=test_set,
  epochs = 20,
  steps_per_epoch=len(train_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt
plt.plot(mod.history['loss'], label='train loss')
plt.plot(mod.history['val_loss'], label='val loss')
plt.legend()
plt.show()

![image](https://user-images.githubusercontent.com/121440744/211757816-5376a342-ef0f-4e14-833b-e1b1691dbe45.png)


plt.plot(mod.history['accuracy'], label='train accuracy')
plt.plot(mod.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

![image](https://user-images.githubusercontent.com/121440744/211757899-b1c61bd0-467e-48a4-8a1e-ea6d4c45029f.png)

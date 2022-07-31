# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:57:36 2022

@author: User
"""
# Import packages
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize empty lists for images and masks from the datasets
images = []
masks=[]
PATH = r'C:\Users\User\OneDrive\Documents\GitHub\ai07-nuclei-semantic-segmentation\datasets\train'

# Load images from datasets using opencv
IMG_PATH = os.path.join(PATH,'inputs')
for image_file in os.listdir(IMG_PATH):
    img = cv2.imread(os.path.join(IMG_PATH,image_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)

# Load masks from datasets
MASK_PATH = os.path.join(PATH,'masks')
for mask_file in os.listdir(MASK_PATH):
    mask = cv2.imread(os.path.join(MASK_PATH,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)

images_np  = np.array(images)
masks_np = np.array(masks)

# display examples of images from datasets
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_np[i])
    plt.axis('off')
plt.show()
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_np[i])
    plt.axis('off')
plt.show()
 
# Data pre-processing

# Add channel 
mask_np_exp = np.expand_dims(masks_np,axis=-1)
print(f"Unique Masks : {np.unique(masks[0])}")
print(f"Masks expanded shape :{mask_np_exp.shape}")

 
cvt_masks = np.round(mask_np_exp/255).astype(np.int64)
print(f"Unique mask converted :{np.unique(cvt_masks[0])}")

cvt_images = images_np/255.0
sample_img = cvt_images[0]


# Train test split
SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(cvt_images,cvt_masks,test_size=0.2,random_state=SEED)

# Convert arrays into tensors
x_train_tensor = tf.data.Dataset.from_tensor_slices(X_train) 
x_test_tensor =tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

# Combine images and masks
train_dataset = tf.data.Dataset.zip((x_train_tensor,y_train_tensor))
test_dataset=  tf.data.Dataset.zip((x_test_tensor,y_test_tensor))

class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='vertical',seed=seed)
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels

# Convert dataset into prefetch dataset
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE/BATCH_SIZE

train_batches = (train_dataset
                 .cache()
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE)
                 .repeat()
                 .map(Augment())
                 .prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_dataset.batch(BATCH_SIZE)

# Visualize images and masks side-by-side
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

# Create image segmentation model

# Transfer learning (MOBILENET)
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

# List of activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False
def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    
    #Apply functional API to construct U-Net
    #Downsampling 
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    # Output layer
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

# Construct UNET with MobileNetV2 as the encoder
OUTPUT_CLASSES = 3
model = unet_model(output_channels=OUTPUT_CLASSES)

# Save structure of model
img_file = 'public/model_plot.png'
plot_model(model, to_file=img_file, show_shapes=True)

# Compile Model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

# Functions to create masks and show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

# Show predictions
show_predictions()

# Train Model
#Hyperparameters for the model
EPOCHS = 10
VAL_SUBSPLITS = 5
VAL_STEPS = len(test_dataset)//BATCH_SIZE//VAL_SUBSPLITS
history = model.fit(train_batches,validation_data=test_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VAL_STEPS)

# Deploy model
print("==================MODEL INFERENCE================================")
show_predictions(test_batches,3)
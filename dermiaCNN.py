# Mikail Usman 
# Dermia AI

import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

# --- Data Pre-processing ---
dataPath = 'C:/Users/m20mi/Documents/Work/Dermia/Dataset'
imgSize = (224, 224) # Setting desired image height and width in pixels

# Function to load individual image (Must process image to match processed dataset)
def loadImage(imgPath):
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR) # Use OpenCV to read image from specified path
    image = cv2.resize(image, imgSize, interpolation=cv2.INTER_AREA) # Image resizing to dataset resolution
    image = img_to_array(image) # Convert image to array
    imgFinal = image.astype('float32')/255.0 # Normalize image
    
    return imgFinal

# Converting the dataset from images in folders into numpy arrays 
def loadData(dataPath, imgSize):
    images = []
    labels = []
    classLabels = os.listdir(dataPath) # Create list of feature/class names (Reads name of each sub-folder within dataset)

    # Enumerate: Adds an index to an iterable (list, etc) and returns it as an object, which can then be used in a for loop
    # Using enumerate object, we use 'label' to access image/sub-folder paths and index to record class in <labels> array
    for index, label in enumerate(classLabels): 
        classFolder = os.path.join(dataPath, label) # [Via Enumerate] Opens contents of selected sub-folder in the dataset.
        # <os.path.isdir> checks if path exists (Boolean)
        if os.path.isdir(classFolder):
            # List of each image in each subfolder via <os.listdir>
            # List is then used to iterate through every image in selected sub-folder
            for img in os.listdir(classFolder): 
                # Resizing and converting image to our needs
                imgPath = os.path.join(classFolder, img) # Full image path
                imgResize = load_img(imgPath, target_size=imgSize) # Resize image to 224,224
                img2Array = img_to_array(imgResize) # Finally converting to numpy array 
                images.append(img2Array) # Add finalized image as array to list of images
                labels.append(index) # Assign class index (correspinding to label) for each image

    return np.array(images), np.array(labels), classLabels

# Verify dataset is processed correctly by looking at first 25 images with labels.
def viewData():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagesTrain[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(classLabels[labelsTrain[i]])
    plt.show()

# --- Creating Model ---
def cnnModel(labelsTrain, labelsTest, imagesTrain, imagesTest):
    ''' Max Pooling is a pooling operation that calculates the maximum value for patches of a feature map, 
        and uses it to create a downsampled (pooled) feature map. 
        It is usually used after a convolutional layer. '''

    # <input_shape>: (image_height, image_width, color_channels) - Channels: Black/White=1, Color:3
    # <layers.Conv2D>: (filters, kernel_size, activation, input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(imgSize[0], (3, 3), activation='relu', input_shape=(imgSize[0], imgSize[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D((imgSize[0]*2), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D((imgSize[0]*2), (3, 3), activation='relu'))

    # Feed output from last layer into the following dense layers (to perform classification)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(9)) # Parameter corresponds to number of classes

    #model.summary() # Display architecture of model (The dimensions tend to shrink as you go deeper in the network)

# --- Compile and Train Model ---

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Batch size kept to a small number if exceeding memory allocation
    history = model.fit(imagesTrain, labelsTrain, batch_size=2, epochs=10, validation_data=(imagesTest, labelsTest))

    return history, model

if __name__ == "__main__":
    # print("TensorFlow version:", tf.__version__)
    # print("GPUs available:", tf.config.list_physical_devices('GPU'))

    img1 = "C:/Users/m20mi/Documents/Work/Dermia/nail-fungus.png"
    img2 = "C:/Users/m20mi/Documents/Work/Dermia/chickenpox.jpg"
    img3 = "C:/Users/m20mi/Documents/Work/Dermia/X.png"
    img4 = "C:/Users/m20mi/Documents/Work/Dermia/larva.png"
    img5 = "C:/Users/m20mi/Documents/Work/Dermia/IMG_5027.png"
    userImages = [img1, img2, img3, img4, img5]

    images, labels, classLabels = loadData(dataPath, imgSize) # Call function loadData() to process all data
    images = images.astype('float32')/255.0 # Normalize images [Scaling pixel values b/w 0 and 1]
    labelsTrain, labelsTest, imagesTrain, imagesTest = train_test_split(labels, images, test_size=0.30)
    history, model = cnnModel(labelsTrain, labelsTest, imagesTrain, imagesTest) # Train model on processed data

# --- Evaluating Trained Model ---
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(imagesTest,  labelsTest, verbose=2)

    print(test_acc*100)

# --- Predicting Classes with Trained Model ---
    classProbability = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    for img in userImages:
        predictions = classProbability.predict()
        print(classLabels[np.argmax(predictions[0])])
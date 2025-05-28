# Mikail Usman 
# Dermia AI

import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.utils import load_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
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
    imgFinal = image/255.0 # Normalize image
    
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
def cnnModel(trainData, valData, classWeightDictionary):
    """
    1. MobileNetV2 is loaded with pretrained weights from ImageNet.
    2. <include_top=False> removes the classification head so we can attach our own.
    3. <trainable=False> freezes the convolutional base to prevent weight updates during training (ideal for small datasets).
    """

    # This function builds, compiles, and trains a CNN using 'Transfer Learning'.
    # Transfer learning with MobileNetV2 (As our dataset has <1000 images per class)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Reduces each feature map to a single number (averages spatial dimensions).
        layers.Dense(128, activation='relu'), # Fully connected layer for learning non-linear combinations of features.
        layers.Dropout(0.3), # Prevents overfitting by randomly dropping 30% of neurons during training.
        layers.Dense(9, activation='softmax') # Output layer with 9 units (for 9 classes) and softmax for multi-class classification.
    ])

    #model.summary() # Display architecture of model (The dimensions tend to shrink as you go deeper in the network)

# --- Compile and Train Model ---

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Batch size kept to a small number if exceeding memory allocation
    history = model.fit(trainData, validation_data=valData, epochs=30, class_weight=classWeightDictionary)

    return history, model

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPUs available:", tf.config.list_physical_devices('GPU'))

    img1 = "C:/Users/m20mi/Documents/Work/Dermia/nail-fungus.png"
    img2 = "C:/Users/m20mi/Documents/Work/Dermia/chickenpox.jpg"
    img3 = "C:/Users/m20mi/Documents/Work/Dermia/X.png"
    img4 = "C:/Users/m20mi/Documents/Work/Dermia/larva.png"
    img5 = "C:/Users/m20mi/Documents/Work/Dermia/IMG_5027.png"
    img6 = "C:/Users/m20mi/Documents/Work/Dermia/Acne_share.png"
    userImages = [img1, img2, img3, img4, img5, img6]

    images, labels, classLabels = loadData(dataPath, imgSize) # Call function loadData() to process all data
    images = images/255.0 # Normalize images [Scaling pixel values b/w 0 and 1]

    # xTrain, xTest, yTrain, yTest -> Data is x while labels are y
    imagesTrain, imagesTest, labelsTrain, labelsTest  = train_test_split(images, labels, test_size=0.30)

# --- Data Augmentation ---
    #  Augment training data to simulate more samples and reduce overfitting.
    valDataGen = ImageDataGenerator()
    datasetAugmentor = ImageDataGenerator(
        rotation_range = 20,
        zoom_range = 0.15,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )
    trainData = datasetAugmentor.flow(imagesTrain, labelsTrain, batch_size=16)
    valData = valDataGen.flow(imagesTest, labelsTest, batch_size=16)

# --- Weight Balancing ---
    # To prevent model bias towards majority classes (If some classes have way more data than others)
    classWeights = compute_class_weight(class_weight=None, classes=np.unique(labelsTrain), y=labelsTrain)
    classWeightDictionary = dict(enumerate(classWeights))

    # Call model on augmented, processed data
    history, model = cnnModel(trainData, valData, classWeightDictionary)

# --- Evaluating Trained Model ---
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(imagesTest,  labelsTest, verbose=2)
    print(f"{test_acc*100}% \n")

# --- Predicting Classes with Trained Model ---
    plt.figure(figsize=(10,10))
    for i, img in enumerate(userImages):
        plt.subplot(1,6,i+1) # Vertical, Horizontal, Index (Subplot shows multiple plots on one figure)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        imgCurrent = loadImage(img)
        plt.imshow(imgCurrent)

        imgCurrent = np.expand_dims(imgCurrent, axis=0) # Add batch dimension to img shape: (224, 224, 3) to (1, 224, 224, 3)
        predictions = model.predict(imgCurrent) # Make the actual prediction using model
        predClass = classLabels[np.argmax(predictions[0])]

        plt.xlabel(predClass)
        print(predClass)
    plt.show()

    model.save('C:/Users/m20mi/Documents/Work/Dermia/Model/dermia_model.h5') # Export model for external use (.h5)
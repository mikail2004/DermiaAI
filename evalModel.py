# Mikail Usman 
# Dermia AI

import numpy as np
from keras.models import load_model
import cv2 # This is 'opencv-python' and not 'opencv-contrib-python'

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/m20mi/Documents/Work/Dermia/Model/dermia_model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/m20mi/Documents/Work/Dermia/Model/labels.txt", "r").readlines()

def classifyImage(imagePath):
    # Processing image
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR) # Use OpenCV to read image from specified path
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) # Resize the raw image into (224-height,224-width) pixels
    cv2.imshow("Input Image", image) # Show the image in a window
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3) # Make the image a numpy array and reshape it to the models input shape.
    image = (image / 127.5) - 1 # Normalize the image array

    # Model predicts class for the given image
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Return prediction and confidence score
    predClass = (class_name[2:]).strip('\n')
    predScore = str(np.round(confidence_score * 100))[:-2] + '%'

    return predClass, predScore

if __name__ == '__main__':
    img1 = "C:/Users/m20mi/Documents/Work/Dermia/nail-fungus.png"
    img2 = "C:/Users/m20mi/Documents/Work/Dermia/chickenpox.jpg"
    img3 = "C:/Users/m20mi/Documents/Work/Dermia/X.png"
    img4 = "C:/Users/m20mi/Documents/Work/Dermia/larva.png"
    img5 = "C:/Users/m20mi/Documents/Work/Dermia/IMG_5027.png"
    img6 = "C:/Users/m20mi/Documents/Work/Dermia/Acne_share.png"

    print(classifyImage(img3))
    cv2.waitKey(0) # Keep image window open until user closes it
    cv2.destroyAllWindows() # Delete all windows from memory once user closes them
# DermiaAI
Dermia AI uses convolutional neural networks (CNNs) to classify images of 9 common skin conditions. Built with TensorFlow and Keras, the model leverages data augmentation and class balancing to improve performance on a moderately sized dataset of 2,078 images. This was accomplished using transfer learning with MobileNetV2. I have included the exported trained model 'dermia_model.5' and an accompanying set of files to utilize it for classification tasks (evalModel.py and labels.txt)

## Dataset Structure
Dataset/
├── Acne/
├── Chicken Pox/
├── Eczema/
├── Impetigo/
├── Larva Migrans/
├── Nail Fungus/
├── Normal/
├── Ringworm/
└── Shingles/

## Requirements
Python 3.10.16
keras 2.10.0
matplotlib 3.10.3
numpy 1.23.5
opencv-python 4.11.0.86
pillow 11.2.1
scikit-learn 1.6.1
tensorflow 2.10.0

import urllib.request
import keras
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from scipy.io import loadmat
import urllib.request
#get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.callbacks import Callback
from PIL import Image
import numpy as np
import yaml
import os
import requests
import tarfile
import scipy.io
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download and extract the image dataset
from tqdm.autonotebook import get_ipython

dataset_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
data_dir = '/home/amitp/Untitled_Folder/path/to/flower_dataset'
# Download and extract the dataset
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    urllib.request.urlretrieve(dataset_url, os.path.join(data_dir, '102flowers.tgz'))
    urllib.request.urlretrieve(labels_url, os.path.join(data_dir, 'imagelabels.mat'))
    tar = tarfile.open(os.path.join(data_dir, '102flowers.tgz'), 'r:gz')
    tar.extractall(data_dir)
    tar.close()
    print("Dataset downloaded and extracted successfully.")


# Step 2: Preprocessing
image_size = (224, 224)  # Resize images to (224, 224) for VGG19 and MobileNetV2

# Load image labels from .mat file
labels = loadmat(os.path.join(data_dir, 'imagelabels.mat'))
labels = labels['labels'][0]


# Split the dataset into training, validation, and test sets
X_train_val_index, X_test_index, y_train_val_index, y_test_index = train_test_split(np.arange(len(labels)), labels, test_size=0.25, random_state=42)
X_train_index, X_val_index, y_train_index, y_val_index = train_test_split(X_train_val_index, y_train_val_index, test_size=0.33, random_state=42)


# Define image preprocessing function
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    return img


num_classes = len(np.unique(labels))


# Preprocess training images
X_train = np.array([preprocess_image(data_dir + "/jpg/image_" + str(i + 1).zfill(5) + ".jpg") for i in X_train_index])
y_train = keras.utils.to_categorical(y_train_index - 1, num_classes=num_classes)

# Preprocess validation images
X_val = np.array([preprocess_image(data_dir + "/jpg/image_" + str(i + 1).zfill(5) + ".jpg") for i in X_val_index])
y_val = keras.utils.to_categorical(y_val_index - 1, num_classes=num_classes)

# Preprocess test images
X_test = np.array([preprocess_image(data_dir + "/jpg/image_" + str(i + 1).zfill(5) + ".jpg") for i in X_test_index])
y_test = keras.utils.to_categorical(y_test_index - 1, num_classes=num_classes)


# ## VGG19

class Callback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []
        self.test_acc = []

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        print(f'\nVGG19 Testing loss: {loss}, accuracy: {acc}\n')
        
test_callback = Callback((X_test, y_test))


# Load pre-trained VGG19 model without the top classification layer
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False


# Add a global average pooling layer and a fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)


predictions = Dense(102, activation="softmax")(x)


#os.system("taskset -p 0x3 {}".format(os.getpid()))


# Create the VGG19 model
vgg19_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
vgg19_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

vgg19_history = vgg19_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=50,
    epochs=5,
    callbacks=[test_callback],
)



# Evaluate VGG19 model
vgg19_scores_loss, vgg19_scores = vgg19_model.evaluate(X_test, y_test)
print("VGG19 Test Accuracy:", vgg19_scores)
print("VGG19 Test Loss:", vgg19_scores_loss)


column_names = vgg19_history.history.keys()
print(column_names)


# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(vgg19_history.history['accuracy'], label='Train')
plt.plot(vgg19_history.history['val_accuracy'], label='Validation')
plt.plot(test_callback.test_acc, label='Test')

plt.title('VGG19 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot cross-entropy loss
plt.figure(figsize=(10, 5))
plt.plot(vgg19_history.history['loss'], label='Train')
plt.plot(vgg19_history.history['val_loss'], label='Validation')
plt.plot(test_callback.test_loss, label='Test')
plt.title('VGG19 Model Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## YOLOv5

#In order to run Yolov5 you need first to clone from git
"""!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt #install"""

# Step 1: Download and extract the image dataset and labels
dataset_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
data_dir = '/home/amitp/Untitled_Folder'

# Function to download a file from the internet
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Download the image dataset and labels
download_file(dataset_url, os.path.join(data_dir, '102flowers.tgz'))
download_file(labels_url, os.path.join(data_dir, 'imagelabels.mat'))

# Extract the image dataset
with tarfile.open(os.path.join(data_dir, '102flowers.tgz'), 'r:gz') as tar:
    tar.extractall(path=data_dir)

# Load the labels
labels = scipy.io.loadmat(os.path.join(data_dir, 'imagelabels.mat'))['labels'][0]

# Split the dataset into training, validation, and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    os.listdir(os.path.join(data_dir, 'jpg')), labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Create the output directories
output_dir = '/home/amitp/Untitled_Folder/datasets/flower_dataset'
os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)


# Function to move files and save labels
def move_files_and_save_labels(images, labels, split, output_dir):
    for image, label in zip(images, labels):
        # Move the image
        dest_path = os.path.join(output_dir, split, 'images', image)
        shutil.move(os.path.join(data_dir, 'jpg', image), dest_path)
        
        # Open the image and get its size
        with Image.open(dest_path) as img:
            img_width, img_height = img.size
        
        # Calculate the center, width and height
        x_center = img_width / 2.0
        y_center = img_height / 2.0
        width = img_width
        height = img_height

        # Normalize the center, width and height values
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # Save the label and coordinates
        with open(os.path.join(output_dir, split, 'labels', image.replace('.jpg', '.txt')), 'w') as f:
            f.write(f"{label} {x_center} {y_center} {width} {height}")


# Move the images and save the labels
move_files_and_save_labels(train_images, train_labels, 'train', output_dir)
move_files_and_save_labels(val_images, val_labels, 'val', output_dir)
move_files_and_save_labels(test_images, test_labels, 'test', output_dir)



# Paths to train, val, and test image folders
train_folder = '/home/amitp/Untitled_Folder/datasets/flower_dataset/train/images'
val_folder = '/home/amitp/Untitled_Folder/datasets/flower_dataset/val/images'
test_folder = '/home/amitp/Untitled_Folder/datasets/flower_dataset/test/images'

# Get unique class labels
unique_labels = np.unique(train_labels)

# Convert numpy scalar to regular Python scalar
def numpy_scalar_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', str(data))

# Register the numpy scalar representer
yaml.add_representer(np.float32, numpy_scalar_representer)

# Create classes dictionary
classes = {idx: label.item() for idx, label in enumerate(unique_labels)}
classes[102] = 103

# Create data dictionary
data = {
    'path': '../datasets/flower_dataset',
    'train': train_folder,
    'val': val_folder,
    'test': test_folder,
    'names': classes
}

# Save data dictionary to YAML file
yaml_path = '/home/amitp/Untitled_Folder/yolov5/data/data.yaml'  # Change the path as needed
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file)

print("Created data.yaml file")


#train task in yolov5
get_ipython().system('python "/home/amitp/Untitled_Folder/yolov5/train.py" --img 640 --epochs 3 --data data.yaml --weights yolov5s.pt')

#val task in yolov5
get_ipython().system('python "/home/amitp/Untitled_Folder/yolov5/val.py" --img 640 --batch 16 --data data.yaml --weights "/home/amitp/Untitled_Folder/yolov5/runs/train/exp11/weights/best.pt"')

#test task in yolov5
get_ipython().system('python "/home/amitp/Untitled_Folder/yolov5/val.py" --task test --img 640 --batch 16 --data data.yaml --weights "/home/amitp/Untitled_Folder/yolov5/runs/train/exp11/weights/best.pt"')


# Load the data from csv
data = pd.read_csv('/home/amitp/Untitled_Folder/yolov5/runs/train/exp11/results.csv')

# Plot the data
plt.figure(figsize=(12,6))

# Assuming the CSV contains 'precision', 'recall' columns. Modify as needed.
plt.plot(data["      train/cls_loss"], label='Train-loss')
plt.plot(data["        val/cls_loss"], label='Validation-loss')

plt.title('Training Results')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()

plt.show()



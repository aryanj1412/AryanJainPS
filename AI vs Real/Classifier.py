import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2, ResNet152, VGG16, EfficientNetB0, InceptionV3
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def createdataframe_test(dir):
    image_paths = []
    for imagename in os.listdir(dir):
        image_paths.append(os.path.join(dir, imagename))  # Add image path
    return image_paths



def extract_features(images):
    features = []
    valid_images = []
    for image in tqdm(images):
        try:
            img = load_img(image, target_size=(236, 236))
            img = np.array(img)
            features.append(img)
            valid_images.append(image)  # Only keep the valid images
        except Exception as e:
            print(f"Skipping image {image} due to error: {e}")
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)
    return features, valid_images



TRAIN_DIR = r"C:\Users\Aryan jain\Downloads\Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

train_features, image = extract_features(train['image'])

x_train = train_features / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

class_labels = le.transform(train['label'])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model.fit(x=x_train, y=y_train, batch_size=32, epochs=5)

TEST_DIR = r"C:\Users\Aryan jain\Downloads\Test"
test = pd.DataFrame()
test['image'] = createdataframe_test(TEST_DIR)

test_features, images = extract_features(test['image'])

x_test = test_features / 255.0

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

predicted_labels = le.inverse_transform(predicted_classes)

test_results = pd.DataFrame({
    'Id': [(os.path.splitext(os.path.basename(image))[0]) for image in images],
    'Label': predicted_labels
})
test_results.sort_values(
    'Id',
    key=lambda x: x.str.extract(r'(\d+)')[0].astype(int),
    inplace=True
)
# Save the results to CSV
test_results.to_csv(r"C:\Users\Aryan jain\Desktop\submission.csv", index=False)

print("Test predictions saved to CSV.")

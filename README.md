# CNN Model for Image Classification

This repository contains a Convolutional Neural Network (CNN) model for image classification using an augmented dataset. The model is built using TensorFlow and Keras, and it leverages ResNet50 for feature extraction.

## Dataset
The model is trained on an augmented dataset that includes transformations such as rotations, flips, and color adjustments to improve generalization.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Data Preprocessing & Augmentation
The dataset is loaded and augmented using `ImageDataGenerator`:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

## Model Architecture
The model utilizes ResNet50 as a feature extractor, followed by fully connected layers for classification.
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input

inputs = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Training
To train the model, run the following script:
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, lr_scheduler]
)
```

## Evaluation
The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

print(classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys()))
```

## Results
The confusion matrix provides insight into classification performance:
```python
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```

## Model Inference
To make predictions on a new image:
```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return train_generator.class_indices.keys()[np.argmax(predictions)]
```

## Saving & Loading the Model
```python
model.save("resnet50_augmented.h5")

# Load model
tf.keras.models.load_model("resnet50_augmented.h5")
```

## License
This project is open-source and available under the MIT License.


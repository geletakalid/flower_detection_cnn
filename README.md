# Flower Classification using Deep Learning

## Project Overview
This project implements a **deep learning model** to classify flower images into **five different categories**. The model is trained on a dataset containing flower images and utilizes a **convolutional neural network (CNN)** for classification.

## Dataset
The dataset consists of:
- **Unaugmented Data**: Raw images without any modifications.
- **Augmented Data**: Processed images with transformations such as **rotation, flipping, and scaling** to enhance model generalization.

## Prerequisites
Ensure you have the following dependencies installed before running the project:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
```

## Model Architecture
The classification model is built using a **CNN (Convolutional Neural Network)** with the following layers:
- **Convolutional layers** with ReLU activation
- **Max pooling layers** for dimensionality reduction
- **Fully connected dense layers**
- **Softmax activation** for multi-class classification

## Training Process
1. Load the dataset (**augmented and unaugmented images**).
2. Preprocess images (**resize, normalize, augment**).
3. Split data into **training and testing sets**.
4. Train the **CNN model**.
5. Evaluate performance using **accuracy and loss metrics**.

## Code Explanation
### `train.py` (Training the Model)
#### Step 1: Import Libraries
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
```
**Explanation:**
- TensorFlow and Keras are used for deep learning.
- `Sequential` is used to create a linear stack of layers.
- `Conv2D` and `MaxPooling2D` are used for feature extraction.
- `Flatten` and `Dense` layers create the final classification structure.
- `ImageDataGenerator` helps in data augmentation.


**Explanation:**
- The image is read using OpenCV.
- The image is resized to match the model input size (150x150 pixels).
- The pixel values are normalized.
- `expand_dims` is used to add an extra dimension for batch processing.

#### Step 4: Make a Prediction
```python
image_path = 'test_flower.jpg'
processed_image = preprocess_image(image_path)
predictions = model.predict(processed_image)
class_index = np.argmax(predictions)
class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
predicted_class = class_labels[class_index]
print(f'Predicted Flower: {predicted_class}')
```
**Explanation:**
- The model predicts the class probabilities.
- `argmax` finds the class with the highest probability.
- The corresponding flower name is retrieved and displayed.

## Usage
Run the following command to **start training**:

```bash
python train.py
```

To **classify a new image**, use:
```bash
python classify.py --image path_to_image.jpg
```

## Evaluation Metrics
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix**

## Results
The trained model achieves **high accuracy** in classifying flower images into **five categories**. Results can be visualized using:
- **Confusion Matrix**
- **Accuracy/Loss plots**

## Future Enhancements
- Implement **transfer learning** using pre-trained models like **VGG16, ResNet**
- Optimize **hyperparameters** for better performance
- Deploy as a **web application** using Flask or FastAPI

## Author
**Geleta Kalid**

## License
This project is **open-source** and available under the **MIT License**.


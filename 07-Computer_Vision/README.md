## Module 7: Computer Vision

Computer Vision is a field of artificial intelligence that enables machines to interpret and understand visual information from the world, similar to the way humans do. It involves techniques to analyze images and videos to extract meaningful insights and make decisions based on that information. This module will cover key concepts in computer vision, accompanied by explanations and examples.

---

### 1. **Image Preprocessing**

Image preprocessing is the first step in many computer vision tasks, such as image classification, object detection, and segmentation. The goal is to improve the image quality and make it easier for a model to extract relevant features.

#### Key tasks in image preprocessing:

- **Grayscale Conversion**: Converting an image to grayscale can simplify the problem, especially for tasks like image classification.
- **Resizing**: Adjusting the image size to fit the input size required by the model.
- **Normalization**: Scaling pixel values to a specific range (usually between 0 and 1) to ensure the model works effectively.
- **Data Augmentation**: Randomly transforming images (rotating, flipping, etc.) to increase the diversity of training data.

#### Example:
```python
import cv2
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('sample_image.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
resized_image = cv2.resize(gray_image, (224, 224))

# Normalize the image
normalized_image = resized_image / 255.0

# Display the images
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale")

plt.subplot(1, 3, 2)
plt.imshow(resized_image, cmap='gray')
plt.title("Resized")

plt.subplot(1, 3, 3)
plt.imshow(normalized_image, cmap='gray')
plt.title("Normalized")

plt.show()
```

**Output:**
This code will display the grayscale, resized, and normalized images for comparison.

---

### 2. **Object Detection**

Object detection involves identifying and locating objects within an image. The goal is to determine not only the presence of an object but also its location using bounding boxes.

#### Common methods for object detection:

- **Haar Cascades**: A classical method used for real-time face detection.
- **YOLO (You Only Look Once)**: A modern, fast, and accurate deep learning-based method.
- **Faster R-CNN**: A region-based convolutional neural network method that performs both object detection and segmentation.

#### Example: Using YOLO for Object Detection
Using a pre-trained YOLO model with OpenCV:
```python
import cv2

# Load YOLO model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load image
img = cv2.imread("image.jpg")
height, width, channels = img.shape

# Prepare image for YOLO model
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo_net.setInput(blob)
outs = yolo_net.forward(output_layers)

# Process the output and draw bounding boxes
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            cv2.rectangle(img, (center_x - w // 2, center_y - h // 2), 
                          (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Output:**
This code detects objects in an image and draws bounding boxes around them using YOLO.

---

### 3. **Image Classification**

Image classification is the task of assigning a label to an image based on its content. It is often the first step in many computer vision tasks and is commonly used in applications like facial recognition, object categorization, and more.

#### Example: Using a Pre-trained Model for Image Classification
Using `TensorFlow` and a pre-trained model (e.g., MobileNet) to classify an image:
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load and preprocess the image
img = image.load_img('sample_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Decode the predictions
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

# Display the image
plt.imshow(img)
plt.title(f"Predicted: {decoded_predictions[0][1]}")
plt.show()
```

**Output:**
This code will output the top 3 predicted labels for the image and display the image with its predicted label.

---

### 4. **Segmentation**

Image segmentation is the task of partitioning an image into multiple segments (regions), making it easier to analyze. These segments can represent various objects or parts of the image, such as foreground and background.

- **Semantic Segmentation**: Classifies each pixel into a category (e.g., car, tree, building).
- **Instance Segmentation**: Differentiates between different objects of the same class.

#### Example: Using Mask R-CNN for Instance Segmentation
Mask R-CNN is a popular model for both object detection and instance segmentation. Here's how you can use it:
```python
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained Mask R-CNN model
model = tf.keras.applications.MaskRCNN()

# Load image
image = cv2.imread('image.jpg')

# Process the image
processed_image = cv2.resize(image, (800, 600))
input_array = np.expand_dims(processed_image, axis=0)

# Perform segmentation
masks = model.predict(input_array)

# Visualize the segmentation masks
for mask in masks:
    plt.imshow(mask, cmap='gray')
    plt.show()
```

**Output:**
This code will display the segmented regions of an image, distinguishing between different instances.

---

### 5. **Generative Adversarial Networks (GANs)**

Generative Adversarial Networks (GANs) consist of two neural networks: a generator and a discriminator. The generator creates fake data, while the discriminator attempts to distinguish between real and fake data. Through this adversarial process, GANs learn to generate data similar to the real-world data they were trained on.

#### Example: Generating Fake Images with GANs
Using `Keras` to implement a simple GAN for generating images:
```python
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
import numpy as np
import matplotlib.pyplot as plt

# Simple generator and discriminator for GAN
def create_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(784, activation='tanh'))
    return model

# Generate random noise as input
noise = np.random.normal(0, 1, (1, 100))

# Create and generate fake image
generator = create_generator()
fake_image = generator.predict(noise).reshape(28, 28)

# Display the generated image
plt.imshow(fake_image, cmap='gray')
plt.title("Generated Fake Image")
plt.show()
```

**Output:**
This code will generate a random fake image (typically similar to handwritten digits if trained on MNIST).

---

### Resources

- **Coursera: Computer Vision Specialization**  
  A detailed series of courses that teach essential computer vision concepts, from basic image processing to advanced techniques like object detection and segmentation.
  
- **Stanford CS231n: Convolutional Neural Networks for Visual Recognition**  
  A popular course on computer vision that covers deep learning techniques for visual recognition, including convolutional neural networks (CNNs) and GANs.
  - Website: [CS231n Course Materials](http://cs231n.stanford.edu/)

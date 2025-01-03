# CNN-on-MNIST-Dataset
Performing CNN model MNIST Dataset 
This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The MNIST dataset is a well-known benchmark in the field of machine learning and computer vision, consisting of grayscale images of digits ranging from 0 to 9.

**Overview**

The goal of this project is to build a CNN that achieves high accuracy in classifying the digits in the MNIST dataset. The project includes steps such as data preprocessing, building the CNN model, training the model, and evaluating its performance.

**Dataset**

The MNIST dataset consists of:

Training Set: 60,000 images

Test Set: 10,000 images

Each image is a 28x28 grayscale image of a single digit (0-9).

The dataset is preprocessed to normalize pixel values to the range [0, 1] for faster convergence during training.

**Model Architecture**

The CNN model consists of the following layers:

Convolutional Layers: Extract spatial features from the input images using filters.

Pooling Layers: Downsample the feature maps to reduce dimensionality and computational load.

Fully Connected Layers: Perform classification based on the extracted features.

**Installation**

To run the notebook and train the model, follow these steps:

Clone the repository:

git clone <[repository_url](https://github.com/NoorUnNisaSoomro/CNN-on-MNIST-Dataset/blob/main/CNN_on_MNIST_dataset.ipynb)>

Navigate to the project directory:

cd cnn_mnist_project

Install the required dependencies:

pip install -r requirements.txt

**Usage**

Open the Jupyter Notebook:

jupyter notebook CNN_on_MNIST_dataset.ipynb

Execute the cells sequentially to:

Load and preprocess the MNIST dataset.

Define the CNN model architecture.

Train the model on the training dataset.

Evaluate the model on the test dataset.

Visualize results such as accuracy and loss.

**Results**

The trained CNN achieves the following metrics on the test dataset:

Accuracy: Approximately 98%

Loss: Low loss, demonstrating good generalization.

**Visualizations include:**

Training and validation accuracy/loss plots.

Examples of correct and incorrect predictions.

**Technologies Used**

Programming Language: Python

Deep Learning Framework: TensorFlow/Keras

Libraries: NumPy, Matplotlib

**Acknowledgments**

The MNIST dataset is provided by Yann LeCun and is available at MNIST Database.

Inspiration and guidance were taken from online resources and the TensorFlow documentation.

Feel free to customize or improve the model and explore additional techniques to further enhance its performance.


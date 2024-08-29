
# Flower Classifications

This project focuses on classifying images of flowers into five categories: **daisy, dandelion, roses, sunflowers**, and **tulips**. Various deep learning models were developed, evaluated, and compared to achieve this classification. The most effective model, EfficientNetB0, is deployed for interactive use.

## Project Structure

- **Flower_classification_CNN.ipynb**: A Convolutional Neural Network (CNN) model designed as a baseline for flower classification.
- **Flower_classification_ResNet.ipynb**: A model based on the ResNet50 architecture, significantly improving classification performance over the baseline.
- **Flower_classification_VGG16.ipynb**: A model based on the VGG16 architecture, offering a mid-level performance between ResNet and EfficientNet.
- **Flower_classification_EfficientNet.ipynb**: The best-performing model based on EfficientNetB0, achieving the highest accuracy among the models evaluated.
- **Deployment**: The EfficientNet model is deployed on Hugging Face using Gradio, providing an interactive interface to classify images of flowers.

## Dataset

The dataset used in this project includes images of the following flower classes:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

Each image is labeled with its corresponding flower class. The dataset is divided into training and testing sets, ensuring that the models are properly validated.

## Models

### 1. CNN Model
- **File**: `Flower_classification_CNN.ipynb`
- **Accuracy**: ~65%
- **Overview**: This model uses a basic CNN architecture with several convolutional layers followed by pooling layers. It serves as a baseline for the flower classification task.

### 2. ResNet Model
- **File**: `Flower_classification_ResNet.ipynb`
- **Accuracy**: 88.10%
- **Overview**: This model leverages the ResNet50 architecture, which allows for deeper networks using residual connections. It significantly outperforms the CNN model.

### 3. VGG16 Model
- **File**: `Flower_classification_VGG16.ipynb`
- **Accuracy**: ~84%
- **Overview**: This model is based on the VGG16 architecture, which uses 16 layers to achieve moderate performance in flower classification.

### 4. EfficientNet Model
- **File**: `Flower_classification_EfficientNet.ipynb`
- **Accuracy**: 93.05%
- **Overview**: This model leverages the EfficientNetB0 architecture, achieving the highest accuracy among all the models. EfficientNetB0 uses a compound scaling method, optimizing width, depth, and resolution in its network architecture.
- **Deployment**: The EfficientNet model has been deployed on Hugging Face using Gradio, providing an interactive interface for testing flower classification.
## Installation and Usage

To run these models locally, you'll need to have Python and the necessary libraries installed. Follow the steps below to set up your environment:

1. **Clone this repository:**

    ```bash
    git clone https://github.com/Pragat007/Flower_Classifications.git
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Open the Jupyter notebooks:**

    ```bash
    jupyter notebook
    ```
4. **Run the cells in any of the notebook files** (`Flower_classification_CNN.ipynb`, `Flower_classification_ResNet.ipynb`, `Flower_classification_VGG16.ipynb`, or `Flower_classification_EfficientNet.ipynb`) **to train and evaluate the models.**

### Access the Deployed App

You can interact with the EfficientNet model directly through the [Hugging Face interface](https://huggingface.co/spaces/Pragat007/Flower_classification), which uses Gradio for a user-friendly experience.

## Results

The EfficientNet model outperformed all other models, achieving an accuracy of 93.05%. This demonstrates the effectiveness of advanced architectures in image classification tasks. The deployment of the EfficientNet model on Hugging Face further enhances accessibility and usability.

## Future Work

Potential improvements and future directions for this project include:
- Implementing data augmentation techniques to improve model generalization.
- Further hyperparameter tuning for better performance.

## Conclusion

This project successfully demonstrates the application of deep learning models in flower classification. The EfficientNet model, in particular, showcases the power of modern architectures in achieving high accuracy in image classification tasks. The deployment on Hugging Face using Gradio further enhances accessibility and usability for end users.

# Flower Classifications

This project focuses on classifying images of flowers into five categories: **daisy**, **dandelion**, **roses**, **sunflowers**, and **tulips**. Two different models were developed and evaluated to achieve this classification.

## Project Structure

- `Flower_classification_CNN.ipynb`: A Convolutional Neural Network (CNN) model designed for flower classification. This model achieved an accuracy of approximately **65%**.
- `Flower_classification_ResNet.ipynb`: A more advanced model based on the ResNet50 architecture, which significantly improved the classification performance, achieving an accuracy of **88.10%**.
- **Deployment**: The ResNet model is deployed on Hugging Face using Gradio, allowing for an interactive interface to classify images of flowers.

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
- **Overview**: This model leverages the ResNet50 architecture, which allows for deeper networks by utilizing residual connections. The ResNet model significantly outperforms the CNN model in this task.
- **Deployment**: The ResNet model has been deployed on Hugging Face using Gradio, providing an interactive interface to test flower classification.

## Installation and Usage

To run these models locally, you'll need to have Python and the necessary libraries installed. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/Pragat007/Flower_Classifications.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter notebooks:
    ```bash
    jupyter notebook
    ```
4. Run the cells in `Flower_classification_CNN.ipynb` or `Flower_classification_ResNet.ipynb` to train and evaluate the models.

### Access the Deployed App

You can interact with the ResNet model directly through the [Hugging Face interface](https://huggingface.co/spaces/Pragat007/Flower_classification), which uses Gradio for a user-friendly experience.

## Results

The ResNet model provided a significant improvement over the CNN model, achieving an accuracy of 88.10%. This demonstrates the effectiveness of deeper architectures with residual connections in image classification tasks.

## Future Work

Potential improvements and future directions for this project include:

- Data augmentation techniques to improve model generalization.
- Hyperparameter tuning for better performance.
- Experimenting with other deep learning architectures like EfficientNet or VGG.

## Conclusion

This project successfully demonstrates the application of deep learning models in flower classification. The ResNet model, in particular, showcases the power of advanced architectures in achieving high accuracy in image classification tasks. The deployment on Hugging Face using Gradio further enhances accessibility and usability.

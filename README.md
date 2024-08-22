# SkinLesionClassificationDL

# Skin Lesion Classification

This repository contains code for training and testing skin lesion classification models using various neural network architectures. The code also includes scripts for cross-validation, feature extraction, and model testing.

## Project Structure

The main directories and files in the project are organized as follows:

/app/
│
├── data/
│ ├── HAM10000_images/ # Directory containing HAM10000 images
│ ├── HAM10000_metadata/ # Directory containing metadata for HAM10000 images
│ ├── ISIC2018_Task3_Test_Images/ # Directory containing test images for ISIC2018 Task 3
│ ├── ISIC2018_Task3_Test_GroundTruth/ # Ground truth for ISIC2018 Task 3 test images
│
├── models/
│ ├── resnet/ # Directory for ResNet model files
│ │ ├── best_model_ham10000_fold_1.pth # Best model weights for ResNet (fold 1)
│ │ ├── best_model_ham10000_fold_2.pth # Best model weights for ResNet (fold 2)
│ │ ├── best_model_ham10000_fold_3.pth # Best model weights for ResNet (fold 3)
│ ├── densenet/ # Directory for DenseNet model files
│ │ ├── best_model_ham10000_fold_1.pth # Best model weights for DenseNet (fold 1)
│ │ ├── best_model_ham10000_fold_2.pth # Best model weights for DenseNet (fold 2)
│ │ ├── best_model_ham10000_fold_3.pth # Best model weights for DenseNet (fold 3)
│ ├── efficientnet/ # Directory for EfficientNet model files
│ │ ├── best_model_ham10000_fold_1.pth # Best model weights for EfficientNet (fold 1)
│ │ ├── best_model_ham10000_fold_2.pth # Best model weights for EfficientNet (fold 2)
│ │ ├── best_model_ham10000_fold_3.pth # Best model weights for EfficientNet (fold 3)
│ ├── extracted_features_train_val.csv # Extracted features from the training/validation set
│
├── training_data/
│ ├── resnet/cross_validation_results_ham10000.csv # Cross-validation results for ResNet
│ ├── densenet/cross_validation_results_ham10000.csv # Cross-validation results for DenseNet
│ ├── efficientnet/cross_validation_results_ham10000.csv # Cross-validation results for EfficientNet
│
├── scripts/
│ ├── HAM10000Dataset.py # Dataset class for loading and processing images
│ ├── ModelTester.py # Script for testing the model on a test dataset
│ ├── SimpleTrainingMetricsPlotter.py # Script for plotting training metrics
│ ├── SkinLesionClassifier.py # Class for handling the skin lesion classification model
│
└── docker/
├── Dockerfile # Dockerfile for setting up the environment
├── requirements.txt # List of Python packages required for the project

### 1. `cross_val_images.ipynb`

- **Purpose**: This notebook handles cross-validation of the models using the training dataset. It evaluates the model's performance across multiple folds and saves the results for further analysis.
- **Key Steps**:
  - Split the dataset into training and validation sets.
  - Perform cross-validation using different architectures (ResNet, DenseNet, EfficientNet).
  - Save the cross-validation metrics for each model architecture.

### 2. `feature_extraction.ipynb`

- **Purpose**: This notebook is used for extracting features from the images using a pre-trained model. The features are then saved to a CSV file for use in downstream tasks such as classification or clustering.
- **Key Steps**:
  - Load the images and apply necessary transformations.
  - Use the `SkinLesionClassifier` to extract features from the dataset.
  - Save the extracted features to a CSV file.

### 3. `test.ipynb`

- **Purpose**: This notebook is used for testing the trained model on a test dataset. It loads the best model weights, processes the test images, and outputs the classification results.
- **Key Steps**:
  - Load test images and metadata.
  - Apply transformations to the test images.
  - Initialize the `SkinLesionClassifier` with a chosen model architecture (e.g., ResNet, DenseNet, EfficientNet).
  - Load the pre-trained model weights.
  - Extract features from the test dataset using the trained model.
  - Save the extracted features to a CSV file.

## How to Use

1. **Setup**:
   - Use the provided `Dockerfile` located in the `/app/docker/` directory to set up the environment. Build the Docker image using the following command:
     
     ```bash
     docker build -t skin-lesion-classifier -f docker/Dockerfile .
     ```

   - Once the image is built, you can run a container with:
   
     ```bash
     docker run -it --rm -v $(pwd):/app skin-lesion-classifier
     ```

   - This will mount the current directory into the Docker container and allow you to run the code in an isolated environment.

2. **Running the Notebooks**:
   - Open each notebook in Jupyter Notebook or JupyterLab.
   - Follow the instructions in the notebooks to execute the cells sequentially.

3. **Training and Testing**:
   - Use the `cross_val_images.ipynb` notebook for cross-validation.
   - Use the `feature_extraction.ipynb` notebook to extract features from images.
   - Use the `test.ipynb` notebook to test the model on the test dataset.


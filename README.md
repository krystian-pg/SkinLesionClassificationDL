# SkinLesionClassificationDL

### 1. `test.ipynb`

- **Purpose**: This notebook is used for testing the trained model on a test dataset. It loads the best model weights, processes the test images, and outputs the classification results.
- **Key Steps**:
  - Load test images and metadata.
  - Apply transformations to the test images.
  - Initialize the `SkinLesionClassifier` with a chosen model architecture (e.g., ResNet, DenseNet, EfficientNet).
  - Load the pre-trained model weights.
  - Extract features from the test dataset using the trained model.
  - Save the extracted features to a CSV file.

### 2. `cross_val_images.ipynb`

- **Purpose**: This notebook handles cross-validation of the models using the training dataset. It evaluates the model's performance across multiple folds and saves the results for further analysis.
- **Key Steps**:
  - Split the dataset into training and validation sets.
  - Perform cross-validation using different architectures (ResNet, DenseNet, EfficientNet).
  - Save the cross-validation metrics for each model architecture.

### 3. `feature_extraction.ipynb`

- **Purpose**: This notebook is used for extracting features from the images using a pre-trained model. The features are then saved to a CSV file for use in downstream tasks such as classification or clustering.
- **Key Steps**:
  - Load the images and apply necessary transformations.
  - Use the `SkinLesionClassifier` to extract features from the dataset.
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
   - Use the `test.ipynb` notebook to test the model on the test dataset.
   - Use the `feature_extraction.ipynb` notebook to extract features from images.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

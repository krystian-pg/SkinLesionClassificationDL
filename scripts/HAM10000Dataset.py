import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class HAM10000Dataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None, augmentations=None, balance_classes=True):
        """
        Args:
            annotations (pd.DataFrame): DataFrame containing the dataset annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied to the images.
            augmentations (callable, optional): Optional augmentations to be applied to the images.
            balance_classes (bool, optional): Whether to balance the classes by assigning weights.
        """
        self.annotations = annotations.copy()  # Create a copy of the annotations to avoid modifying the original DataFrame
        self.root_dir = root_dir
        self.transform = transform
        self.augmentations = augmentations

        # Encode labels as integers
        self.label_encoder = LabelEncoder()
        self.annotations.loc[:, 'label_encoded'] = self.label_encoder.fit_transform(self.annotations['dx'])

        # Calculate weights if balancing classes
        if balance_classes:
            self.weights = self._calculate_weights()
        else:
            self.weights = None

    def _calculate_weights(self):
        # Calculate weights based on class frequency for handling imbalances
        class_counts = self.annotations['label_encoded'].value_counts()
        class_weights = 1. / class_counts
        weights = self.annotations['label_encoded'].map(class_weights)
        return torch.DoubleTensor(weights.values)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the image ID and diagnosis (encoded label)
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_id'] + ".jpg")
        image = Image.open(img_name)
        label = self.annotations.iloc[idx]['label_encoded']  # The encoded label

        # Apply augmentations if they exist (only for training)
        if self.augmentations:
            image = self.augmentations(image)

        # Apply basic transformations (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        # Convert label to a tensor
        label = torch.tensor(label).long()  # Ensure the label is a tensor of type long for classification

        return image, label


    def show_examples(self, num_images=5):
        """Show a few example images with and without augmentation."""
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
        for i in range(num_images):
            idx = np.random.randint(0, len(self))
            
            # Get the original image without any augmentations
            img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_id'] + ".jpg")
            img_original = Image.open(img_path)
            label = self.annotations.iloc[idx]['dx']  # Original label for display
            
            if self.transform:
                img_original = self.transform(img_original)
            
            # Get the augmented image
            img_augmented, _ = self[idx]  # This calls __getitem__, applying both transform and augmentation
            
            # Undo transformations for display (assumes normalization was last)
            img_original_display = img_original.permute(1, 2, 0).numpy()  # Move channels to the last dimension
            img_original_display = img_original_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_original_display = np.clip(img_original_display, 0, 1)

            img_augmented_display = img_augmented.permute(1, 2, 0).numpy()
            img_augmented_display = img_augmented_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_augmented_display = np.clip(img_augmented_display, 0, 1)

            # Show original image
            axes[i, 0].imshow(img_original_display)
            axes[i, 0].set_title(f"Original Image - Label: {label}")
            axes[i, 0].axis('off')

            # Show augmented image
            axes[i, 1].imshow(img_augmented_display)
            axes[i, 1].set_title(f"Augmented Image - Label: {label}")
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def show_class_distribution(self):
        """Show the distribution of classes in the dataset."""
        # Set the seaborn theme for aesthetics
        sns.set_theme(style="whitegrid")

        # Create the plot
        plt.figure(figsize=(12, 7))
        
        # Generate the count plot
        ax = sns.countplot(x='dx', data=self.annotations, 
                           order=self.annotations['dx'].value_counts().index, 
                           palette='colorblind')

        # Add labels on the bars
        for p in ax.patches:
            height = int(p.get_height())  # Convert height to integer
            ax.annotate(f'{height}', 
                        xy=(p.get_x() + p.get_width() / 2., height), 
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points", 
                        ha='center', fontsize=12, color='black')

        # Customize the plot
        ax.set_title('Class Distribution in HAM10000 Dataset', fontsize=16, weight='bold')
        ax.set_xlabel('Class', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Remove the spines to make the plot cleaner
        sns.despine(left=True, bottom=True)
        
        plt.tight_layout()
        plt.show()

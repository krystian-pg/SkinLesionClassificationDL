import torch.nn as nn
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import torch
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights

class SkinLesionClassifier:
    def __init__(self, model_name="resnet50", num_classes=7, weights=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.weights = weights
        self.model = self._get_pretrained_model()
        self.target_layer = self._get_target_layer()

    def _get_pretrained_model(self):
        if self.model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if self.weights else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == "densenet121":
            weights = DenseNet121_Weights.IMAGENET1K_V1 if self.weights else None
            model = models.densenet121(weights=weights)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        elif self.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if self.weights else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Model {self.model_name} is not supported. Choose from 'resnet50', 'densenet121', 'efficientnet_b0'.")
        
        return model

    def _get_target_layer(self):
        """Get the target layer for generating CAM."""
        if self.model_name == "resnet50":
            return self.model.layer4[-1]
        elif self.model_name == "densenet121":
            return self.model.features[-1]
        elif self.model_name == "efficientnet_b0":
            return self.model.features[-1]
        else:
            raise ValueError(f"Model {self.model_name} does not have a defined target layer.")

    def get_model(self):
        return self.model

    def save_model_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        return self.model

    def generate_cam(self, input_tensor, target_category=None):
        """
        Generates Class Activation Maps (CAM) for the input image.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor.
            target_category (int, optional): The target category for generating CAM. Defaults to None.

        Returns:
            np.ndarray: The CAM overlayed on the input image.
        """
        cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category) if target_category else None])
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def overlay_cam_on_image(self, img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays the CAM on the original image.
        
        Args:
            img (np.ndarray): The original image.
            cam (np.ndarray): The CAM to overlay.
            alpha (float): The alpha blending factor.

        Returns:
            np.ndarray: The image with the CAM overlay.
        """
        cam_image = show_cam_on_image(img, cam, use_rgb=True)
        
        # Ensure both images are in the same type
        img = (img * 255).astype(np.uint8)  # Convert img to uint8
        cam_image = (cam_image * 255).astype(np.uint8)  # Convert cam_image to uint8
        
        # Perform the overlay
        return cv2.addWeighted(cam_image, alpha, img, 1 - alpha, 0)

    def extract_features(self, data_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
        """
        Extracts deep features from the model for a given DataLoader.
        
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader containing the data.
            device (torch.device): Device on which to perform the computation (e.g., 'cuda' or 'cpu').

        Returns:
            torch.Tensor: A tensor containing the extracted features for all inputs in the DataLoader.
        """
        self.model.eval()
        features = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                if self.model_name.startswith("resnet"):
                    # Extract features for ResNet models
                    x = self.model.conv1(inputs)
                    x = self.model.bn1(x)
                    x = self.model.relu(x)
                    x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    x = self.model.avgpool(x)
                    x = torch.flatten(x, 1)
                elif self.model_name.startswith("densenet"):
                    # Extract features for DenseNet models
                    x = self.model.features(inputs)
                    x = torch.relu(x)
                    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
                elif self.model_name.startswith("efficientnet"):
                    # Extract features for EfficientNet models
                    x = self.model.features(inputs)
                    x = self.model.avgpool(x)
                    x = torch.flatten(x, 1)
                else:
                    raise ValueError(f"Model {self.model_name} is not supported.")
                
                features.append(x.cpu())
        return torch.cat(features, dim=0)
    
import torch
from sklearn.metrics import classification_report
import pandas as pd

class ModelTester:
    def __init__(self, model_class, test_dataset, num_classes=7, device=None):
        """
        Initialize the tester with the model class and the test dataset.

        Args:
            model_class (class): The model class (e.g., SkinLesionClassifier).
            test_dataset (torch.utils.data.Dataset): The test dataset.
            num_classes (int): Number of classes in the dataset.
            device (torch.device, optional): Device to perform computation on. Defaults to CUDA if available.
        """
        self.model_class = model_class
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _test_single_fold(self, model, test_loader):
        """
        Test a single fold of the model.

        Args:
            model (torch.nn.Module): The trained model.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Classification report as a dictionary.
        """
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        return classification_report(y_true, y_pred, output_dict=True, target_names=[f"Class {i}" for i in range(self.num_classes)])

    def test_all_folds(self, best_models, batch_size=32):
        """
        Test all models across all folds.

        Args:
            best_models (list): List of paths to the best model states for each fold.
            batch_size (int): Batch size for DataLoader.

        Returns:
            tuple: (reports_per_fold, mean_report)
                reports_per_fold (list): List of classification reports (dictionaries) for each fold.
                mean_report (pd.DataFrame): DataFrame containing mean results across all folds.
        """
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        reports_per_fold = []

        for i, model_state in enumerate(best_models):
            print(f"Testing Fold {i+1}")

            # Determine the correct model architecture based on the state_dict being loaded
            state_dict = torch.load(model_state)
            if "layer1.0.conv1.weight" in state_dict:  # This is a ResNet
                model_name = "resnet50"
            elif "features.0.0.weight" in state_dict:  # This is EfficientNet
                model_name = "efficientnet_b0"
            elif "features.norm5.weight" in state_dict:  # This is DenseNet
                model_name = "densenet121"
            else:
                raise ValueError("Unknown model architecture in state_dict")

            # Initialize the classifier with the correct model architecture
            classifier = self.model_class(model_name=model_name, num_classes=self.num_classes, weights=False)
            model = classifier.load_model_weights(model_state)
            model.to(self.device)
            
            # Test the model and get the classification report
            report = self._test_single_fold(model, test_loader)
            reports_per_fold.append(report)

        # Calculate mean results across all folds
        mean_report = self._compute_mean_report(reports_per_fold)

        return reports_per_fold, mean_report

    def _compute_mean_report(self, reports_per_fold):
        """
        Compute the mean classification report across all folds.

        Args:
            reports_per_fold (list): List of classification reports (dictionaries) for each fold.

        Returns:
            pd.DataFrame: DataFrame containing mean results across all folds.
        """
        report_df = pd.DataFrame()

        for report in reports_per_fold:
            fold_report_df = pd.DataFrame(report).transpose()
            report_df = report_df.add(fold_report_df, fill_value=0)

        # Averaging the reports across folds
        report_df /= len(reports_per_fold)

        return report_df

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
from torch import nn
from SkinLesionClassifier import SkinLesionClassifier
from HAM10000Dataset import HAM10000Dataset

class CrossValidationTrainer:
    def __init__(self, dataset, model_name="resnet50", num_classes=1, criterion=None, num_folds=5, batch_size=16, num_epochs=20, lr=0.001, patience=5, train_transform=None, val_transform=None, augmentations=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss as the default criterion for classification
        
        self.dataset = dataset
        self.model_name = model_name
        self.num_classes = num_classes
        self.criterion = criterion
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.augmentations = augmentations
        print(f"Using device: {self.device}")

    def train_fold(self, train_loader, val_loader):
        # Initialize the model using the SkinLesionClassifier class
        classifier = SkinLesionClassifier(model_name=self.model_name, num_classes=self.num_classes, weights=True)
        model = classifier.get_model()
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        epoch_data = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            model.train()
            train_loss_sum = 0.0
            correct_train = 0
            total_train = 0

            for batch in tqdm(train_loader, desc="Training"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = train_loss_sum / len(train_loader)
            train_accuracy = correct_train / total_train
            epoch_data['train_loss'].append(train_loss)
            epoch_data['train_accuracy'].append(train_accuracy)

            print(f"Epoch {epoch+1} Training: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

            # Validation step
            model.eval()
            val_loss_sum = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss_sum += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = val_loss_sum / len(val_loader)
            val_accuracy = correct_val / total_val
            epoch_data['val_loss'].append(val_loss)
            epoch_data['val_accuracy'].append(val_accuracy)

            print(f"Epoch {epoch+1} Validation: Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            # Update the best model if the validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()  # Save the best model state for this fold
                patience_counter = 0
                print('Best model updated!')
            else:
                patience_counter += 1

            # Adjust learning rate and optionally log it
            scheduler.step(val_loss)
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')  # Log the current learning rate

            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_val_loss:.4f}")
                break

        return epoch_data, best_model_state

    def cross_validate(self):
        kfold = KFold(n_splits=self.num_folds, shuffle=True)
        fold_results = []
        best_models = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset.annotations)):
            print(f"FOLD {fold+1}/{self.num_folds}")
            train_subset = self.dataset.annotations.iloc[train_idx]
            val_subset = self.dataset.annotations.iloc[val_idx]

            # Create new datasets for each fold
            train_dataset = HAM10000Dataset(
                annotations=train_subset, 
                root_dir=self.dataset.root_dir, 
                transform=self.train_transform, 
                augmentations=self.augmentations, 
                balance_classes=False  # No need to recalculate weights within the subset
            )
            val_dataset = HAM10000Dataset(
                annotations=val_subset, 
                root_dir=self.dataset.root_dir, 
                transform=self.val_transform, 
                augmentations=None, 
                balance_classes=False  # No need to recalculate weights within the subset
            )

            # Use WeightedRandomSampler for the training set
            if self.dataset.weights is not None:
                train_sampler = WeightedRandomSampler(weights=self.dataset.weights[train_idx], num_samples=len(train_idx), replacement=True)
            else:
                train_sampler = None

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

            epoch_data, best_model_state = self.train_fold(train_loader, val_loader)
            epoch_data['fold'] = fold + 1  # Add fold number to the epoch data

            fold_results.append(pd.DataFrame(epoch_data))
            best_models.append(best_model_state)  # Store the best model state for this fold

        # Combine all fold results into a single DataFrame
        results_df = pd.concat(fold_results, ignore_index=True)
        return results_df, best_models

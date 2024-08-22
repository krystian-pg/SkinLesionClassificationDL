import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class SimpleTrainingMetricsPlotter:
    def __init__(self, metrics_df: pd.DataFrame):
        """
        Initialize the plotter with the metrics dataframe.
        The dataframe should include columns: train_loss, train_accuracy, val_loss, val_accuracy, fold.
        """
        self.metrics_df = metrics_df
        self.metrics_df['epoch'] = self.metrics_df.groupby('fold').cumcount() + 1

    def compute_mean_std(self) -> pd.DataFrame:
        """
        Compute the mean and standard deviation of loss and accuracy for training and validation data by epoch across all folds.
        """
        return self.metrics_df.groupby('epoch').agg(
            mean_train_loss=('train_loss', 'mean'),
            std_train_loss=('train_loss', 'std'),
            mean_train_accuracy=('train_accuracy', 'mean'),
            std_train_accuracy=('train_accuracy', 'std'),
            mean_val_loss=('val_loss', 'mean'),
            std_val_loss=('val_loss', 'std'),
            mean_val_accuracy=('val_accuracy', 'mean'),
            std_val_accuracy=('val_accuracy', 'std')
        ).reset_index()

    def plot_metrics(self) -> None:
        """
        Plot the mean and standard deviation of loss and accuracy by epoch for both training and validation data across all folds.
        """
        metrics = self.compute_mean_std()
        
        sns.set_theme(style="whitegrid", context="talk")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

        # Customize line styles and colors
        line_styles = {
            'train': {'color': 'royalblue', 'label': 'Training', 'marker': 'o'},
            'val': {'color': 'firebrick', 'label': 'Validation', 'marker': 'o'}
        }

        # Plot Loss
        sns.lineplot(
            x='epoch', 
            y='mean_train_loss', 
            data=metrics, 
            ax=axes[0],
            **line_styles['train']
        )
        axes[0].fill_between(
            metrics['epoch'], 
            metrics['mean_train_loss'] - metrics['std_train_loss'], 
            metrics['mean_train_loss'] + metrics['std_train_loss'], 
            alpha=0.3, color=line_styles['train']['color']
        )
        
        sns.lineplot(
            x='epoch', 
            y='mean_val_loss', 
            data=metrics, 
            ax=axes[0],
            **line_styles['val']
        )
        axes[0].fill_between(
            metrics['epoch'], 
            metrics['mean_val_loss'] - metrics['std_val_loss'], 
            metrics['mean_val_loss'] + metrics['std_val_loss'], 
            alpha=0.3, color=line_styles['val']['color']
        )

        axes[0].set_title("Loss by Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(title="Data Type")

        # Plot Accuracy
        sns.lineplot(
            x='epoch', 
            y='mean_train_accuracy', 
            data=metrics, 
            ax=axes[1],
            **line_styles['train']
        )
        axes[1].fill_between(
            metrics['epoch'], 
            metrics['mean_train_accuracy'] - metrics['std_train_accuracy'], 
            metrics['mean_train_accuracy'] + metrics['std_train_accuracy'], 
            alpha=0.3, color=line_styles['train']['color']
        )
        
        sns.lineplot(
            x='epoch', 
            y='mean_val_accuracy', 
            data=metrics, 
            ax=axes[1],
            **line_styles['val']
        )
        axes[1].fill_between(
            metrics['epoch'], 
            metrics['mean_val_accuracy'] - metrics['std_val_accuracy'], 
            metrics['mean_val_accuracy'] + metrics['std_val_accuracy'], 
            alpha=0.3, color=line_styles['val']['color']
        )

        axes[1].set_title("Accuracy by Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(title="Data Type")

        # Show the plot
        plt.show()

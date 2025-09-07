"""
Deep Learning Training Script for CIFAR-10 Image Classification
Author: CODTECH Internship
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from models import CIFAR10CNN
from utils import set_seed, plot_training_curves, plot_confusion_matrix, save_sample_predictions

class CIFAR10Trainer:
    """CIFAR-10 Image Classification Trainer"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()

        # Set random seeds for reproducibility
        set_seed(config['seed'])

        # Initialize model, data loaders, optimizer, etc.
        self.model = CIFAR10CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders()

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        # CIFAR-10 class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_data_loaders(self):
        """Create data loaders for CIFAR-10"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # No augmentation for validation/test
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Download and load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=train_transform
        )

        # Split training set into train and validation
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            trainset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        testset = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=test_transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=self.config['batch_size'], 
            shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.config['batch_size'], 
            shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            testset, batch_size=self.config['batch_size'], 
            shuffle=False, num_workers=2
        )

        self.logger.info(f"Train samples: {len(train_subset)}")
        self.logger.info(f"Validation samples: {len(val_subset)}")
        self.logger.info(f"Test samples: {len(testset)}")

        return train_loader, val_loader, test_loader

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self):
        """Full training loop"""
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step()

            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint('models/best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config['patience']:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    def evaluate(self):
        """Evaluate model on test set"""
        self.logger.info("Evaluating on test set...")

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        self.logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Generate detailed classification report
        report = classification_report(
            all_targets, all_predicted, 
            target_names=self.classes, 
            output_dict=True
        )

        # Save metrics
        metrics = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'classification_report': report
        }

        with open('outputs/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        return all_targets, all_predicted, test_acc

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.logger.info(f"Checkpoint loaded: {path}")

    def generate_visualizations(self, targets=None, predictions=None):
        """Generate all visualizations"""
        # Create outputs directory
        Path('outputs').mkdir(exist_ok=True)

        # Training curves
        plot_training_curves(self.history, 'outputs/training_curves.png')

        if targets is not None and predictions is not None:
            # Confusion matrix
            plot_confusion_matrix(targets, predictions, self.classes, 'outputs/confusion_matrix.png')

            # Sample predictions
            save_sample_predictions(
                self.model, self.test_loader, self.classes, 
                self.device, 'outputs/sample_predictions.png'
            )

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', 
                       help='Model path for evaluation')

    args = parser.parse_args()

    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'patience': 10,
        'seed': 42
    }

    # Create trainer
    trainer = CIFAR10Trainer(config)

    if args.evaluate:
        # Load model and evaluate
        if Path(args.model_path).exists():
            trainer.load_checkpoint(args.model_path)
            targets, predictions, test_acc = trainer.evaluate()
            trainer.generate_visualizations(targets, predictions)
        else:
            print(f"Model not found: {args.model_path}")
    else:
        # Train model
        trainer.train()

        # Evaluate
        targets, predictions, test_acc = trainer.evaluate()

        # Generate visualizations
        trainer.generate_visualizations(targets, predictions)

if __name__ == "__main__":
    main()

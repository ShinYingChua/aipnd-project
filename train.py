import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

class ImageClassifierTrainer:
    def __init__(self):
        self.parse_args()
        self.load_data()
        self.build_model()
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.args.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.gpu else "cpu")

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Train a new network on a dataset")
        parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
        parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
        parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (e.g., vgg16)')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
        parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
        self.args = parser.parse_args()

    def load_data(self):
        train_dir = self.args.data_dir + '/train'
        valid_dir = self.args.data_dir + '/valid'
        test_dir = self.args.data_dir + '/test'

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
            'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
            'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        }

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=64, shuffle=True),
            'valid': torch.utils.data.DataLoader(self.image_datasets['valid'], batch_size=64),
            'test': torch.utils.data.DataLoader(self.image_datasets['test'], batch_size=64),
        }

    def build_model(self):
        self.model = getattr(models, self.args.arch)(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        classifier_input_size = self.model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(classifier_input_size, self.args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.args.hidden_units, len(self.image_datasets['train'].classes)),
            nn.LogSoftmax(dim=1)
        )
        self.model.classifier = classifier

    def train(self):
        num_epochs = self.args.epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs} - Training...")
            self.model.train()
            running_loss = 0
            for inputs, labels in self.dataloaders['train']:
                print("Processing batch...")
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Training loss
                running_loss += loss.item()

            # Validate the model
            self.model.eval()
            with torch.no_grad():
                validation_loss = 0
                correct = 0
                total = 0
                for inputs, labels in self.dataloaders['valid']:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    # Validation loss
                    loss = self.criterion(outputs, labels)
                    validation_loss += loss.item() 
                    # Validation accuracy 
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate average losses and accuracy
            train_loss = running_loss / len(self.dataloaders['train'])
            valid_loss = validation_loss / len(self.dataloaders['valid'])
            valid_accuracy = correct / total * 100 

            # Print the training/validation statistics
            print(f"Epoch {epoch+1}/{self.args.epochs}.. "
                f"Training Loss: {train_loss:.3f}.. "
                f"Validation Loss: {valid_loss:.3f}.. "
                f"Validation Accuracy: {valid_accuracy:.2f}%")
            
        print("Training finished.")

    def save_checkpoint(self):
        save_dir = self.args.save_dir
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        checkpoint_path = f'{save_dir}/checkpoint.pth'
        checkpoint = {
            'model': self.model,
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx,
        }
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    trainer = ImageClassifierTrainer()
    trainer.train()
    trainer.save_checkpoint()

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.models as models

class ModelTrainer:
    """
    Handles training, validation, and evaluation of a ResNet18 model for image classification.

    Responsibilities:
    - Train the model on training data
    - Validate and tune using validation set
    - Evaluate on test set and show metrics
    """

    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = None

    def build_model(self, learning_rate=1e-4):
        """
        Loads a pretrained ResNet18 and prepares it for the classification task.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 4)
        self.model = model.to(torch.device("cpu"))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs=10):
        """
        Trains the model using the training data.

        Args:
            num_epochs (int): Number of training epochs.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call `build_model()` first.")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for images, labels in self.train_loader:
                images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_acc = self.validate()
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss:.4f} - Val Acc: {val_acc:.4f}")

    def validate(self):
        """
        Evaluates model accuracy on the validation set.

        Returns:
            float: Validation accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def evaluate(self):
        """
        Evaluates the model on the test set and computes metrics.

        Returns:
            Tuple: (accuracy, precision, recall, f1_score, y_true, y_pred)
        """
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(torch.device("cpu"))
                labels = labels.to(torch.device("cpu"))
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        return accuracy, precision, recall, f1, y_true, y_pred


    def show_metrics(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Displays classification report and confusion matrix.
        
        Args:
            y_true (List[int]): True labels.
            y_pred (List[int]): Predicted labels.
            class_names (List[str], optional): Names for classes.
            save_path (str or Path, optional): Path to save the confusion matrix image.
        """
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    def run(self, num_epochs=10):
        """
        Runs the full pipeline: build, train, validate, evaluate and show metrics.
        
        Args:
            num_epochs (int): Number of epochs for training.
        """
        print("Building the model...")
        self.build_model()

        print("Starting training...")
        self.train(num_epochs=num_epochs)

        print("Evaluating on test set...")
        y_true, y_pred = self.evaluate()

        print("Showing metrics and confusion matrix...")
        class_names = ["Class 3", "Class 4", "Class 5", "Others"]
        save_path = "/home/leticia/projetos/Avaliacao-MOBIT/classificador-tipos-de-carros/data/processed/confusion_matrix.png"
        self.show_metrics(y_true, y_pred, class_names=class_names, save_path=save_path)
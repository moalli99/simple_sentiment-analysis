import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data_loader, device, class_labels=['Negative', 'Positive']):
    """
    Evaluate a PyTorch model on a given dataset.
    
    Args:
        model: trained PyTorch model
        data_loader: DataLoader containing the dataset to evaluate
        device: torch device ('cpu' or 'cuda')
        class_labels: list of class names for confusion matrix display
        
    Returns:
        accuracy, classification report string, confusion matrix
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Initialize hidden state
            h = model.init_hidden(inputs.size(0), device)
            if isinstance(h, tuple):  # LSTM returns (h, c)
                h = tuple([each.data for each in h])
            else:  # GRU returns h only
                h = h.data
            
            output, _ = model(inputs, h)
            output = output.squeeze().cpu().numpy()
            labels = labels.cpu().numpy()
            
            y_true.extend(labels)
            y_pred.extend(output)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    y_pred_round = np.round(y_pred)
    accuracy = accuracy_score(y_true, y_pred_round)
    report = classification_report(y_true, y_pred_round)
    cm = confusion_matrix(y_true, y_pred_round)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    

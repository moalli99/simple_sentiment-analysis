from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_evaluate_logistic(X_train, y_train, X_val, y_val, max_features=10000, max_iter=1000):
    """
    Train a Logistic Regression model on text data with TF-IDF features
    and evaluate on training and validation sets.
    """
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Initialize and train Logistic Regression
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_vec, y_train)

    # Predict on train and val
    y_train_pred = model.predict(X_train_vec)
    y_val_pred = model.predict(X_val_vec)

    # Compute metrics
    def compute_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, report, cm

    train_acc, train_report, train_cm = compute_metrics(y_train, y_train_pred)
    val_acc, val_report, val_cm = compute_metrics(y_val, y_val_pred)

    # Print metrics
    print("Training Accuracy:", train_acc)
    print("\nTraining Classification Report:\n", train_report)
    print("Validation Accuracy:", val_acc)
    print("\nValidation Classification Report:\n", val_report)

    # Plot confusion matrix for validation set
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=['negative', 'positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix')
    plt.show()



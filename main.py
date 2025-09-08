import torch
from torch import nn, optim
from src.model_base_line_1 import SentimentLSTM
from src.model_base_line_2 import SentimentGRU
from src.model_base_line_3 import SentimentLSTM_Word2Vec
from src.model_base_line_0 import train_evaluate_logistic
from src.train_eval import train_model, eval_model
from src.test import evaluate_model
from data.get_data import get_loaders, get_data_sklearn
from utils.early_stoping import EarlyStopping
from utils.processing import buile_embedding_matrix
import yaml
import matplotlib.pyplot as plt

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, DEVICE, CLIP, EPOCHS, PATH, early_config):
    epoch_train_losses, epoch_train_acc, epoch_val_losses, epoch_val_acc = [], [], [], []

    early_stopping = EarlyStopping(
        patience=early_config["patience"], 
        delta=early_config["delta"], 
        path=PATH
    )

    for epoch in range(EPOCHS):
        # Training
        epoch_loss, epoch_acc, h = train_model(model, train_loader, optimizer, criterion, DEVICE, CLIP)
        epoch_train_losses.append(epoch_loss)
        epoch_train_acc.append(epoch_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}.. Train loss: {epoch_loss:.4f}.. Train acc: {epoch_acc:.4f}")

        # Validation
        h_val = model.init_hidden(train_loader.batch_size, DEVICE)
        val_loss, val_acc, _ = eval_model(model, val_loader, criterion, DEVICE)
        epoch_val_losses.append(val_loss)
        epoch_val_acc.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}.. Val loss: {val_loss:.4f}.. Val acc: {val_acc:.4f}")
        print("-" * 50)

        # Early stopping
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break


    return epoch_train_losses, epoch_train_acc, epoch_val_losses, epoch_val_acc

def load_model(model, PATH, DEVICE):
    state_dict = torch.load(PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    print(f"Model loaded from {PATH}")
    return model

def plot_loss_acc(train_losses, val_losses, train_acc, val_acc):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label="Train acc")
    plt.plot(val_acc, label="Val acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load config
    with open("config.yaml","r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters from config
    #loader parameters
    file_path = config["file_path"]
    vocab_size = config["vocab_size"]
    seq_len = config["SEQ_LEN"]
    batch_size = config["BATCH_SIZE"]
    temp_size = config["temp_size"]
    test_size = config["test_size"]
    random_state = config["random_state"]
    #model parameters
    EMBEDDING_DIM = config["EMBEDDING_DIM"]
    HIDDEN_DIM = config["HIDDEN_DIM"]
    N_LAYERS = config["N_LAYERS"]
    OUTPUT_DIM = config["OUTPUT_DIM"]
    DROPOUT = config["DROPOUT"]
    CLIP = config["CLIP"]
    #training parameters
    LR = config["LR"]
    EPOCHS = config["EPOCHS"]
    #model selection
    model_base_line = config["model_base_line"]
    #


    # Baseline Logistic Regression
    if config['model_base_line'] == 0:
        X_train, y_train, X_val, y_val = get_data_sklearn()
        train_evaluate_logistic(X_train, y_train, X_val, y_val)
        exit()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader, word_dic = get_loaders(
        file_path,
        vocab_size,
        seq_len,
        batch_size,
        temp_size,
        test_size,
        random_state
    )
    vocab_size = len(word_dic) + 1

    # Model selection
    PATH = ""
    if model_base_line == 1:
        model = SentimentLSTM(N_LAYERS, vocab_size, HIDDEN_DIM, EMBEDDING_DIM, OUTPUT_DIM, DROPOUT)
        PATH = "model_base_line_1.pt"
    elif model_base_line == 2:
        model = SentimentGRU(N_LAYERS, vocab_size, HIDDEN_DIM, EMBEDDING_DIM, OUTPUT_DIM, DROPOUT)
        PATH = "model_base_line_2.pt"
    else:
        #Glove path
        glove_path=config["Glove_path"]
        # Build embedding matrix
        embedding_matrix=buile_embedding_matrix(word_dic,glove_path,EMBEDDING_DIM)
        model = SentimentLSTM_Word2Vec(embedding_matrix,HIDDEN_DIM,N_LAYERS,OUTPUT_DIM,DROPOUT)
        PATH = "model_base_line_3.pt"


    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training & Validation
    train_losses, train_acc, val_losses, val_acc = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, DEVICE, CLIP, EPOCHS, PATH, config["early_stopping"]
    )
    # Plot loss and accuracy
    plot_loss_acc(train_losses, val_losses, train_acc, val_acc)

    # Load best model
    model = load_model(model, PATH, DEVICE)

    # Final evaluation on test set
    evaluate_model(model, test_loader, DEVICE)

   

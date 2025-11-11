import torch
import numpy as np
import sklearn
import torch.nn as nn
from model import model

def TrainingAlgoritm():
    input_data, trues = torch.load('input_data.pt'), torch.load('trues')
    training_model = model()
    return None #Skal vel ikke returnere noe

def LossFunc(y_pred:torch, y:torch)->float:
    '''can use 
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss()
    
    During training: loss = criterion(y_pred, y)'''
    return None 

def get_data(filename: str)->torch: #Will likely not use like this, but just a start
    data = torch.load(filename)
    return data


''' From chat '''

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import model 

def load_data(path):
    obj = torch.load(path, map_location="cpu")
    # Accept dict or tuple
    if isinstance(obj, dict):
        X, y = obj["X"], obj["y"]
    elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
        X, y = obj[0], obj[1]
    else:
        raise ValueError("Unsupported data format in .pt file")
    # Ensure float, correct shapes
    if X.ndim > 2:
        X = X.view(X.size(0), -1)  # (N, M*M)
    X = X.float()
    y = y.float().view(-1, 1)     # (N, 1) for BCELoss with sigmoid output
    return X, y

def main(
    data_path="data/train.pt",
    weights_path="models/model_weights.pth",
    batch_size=256,
    lr=1e-3,
    epochs=20,
    val_fraction=0.1,
    seed=42
):
    torch.manual_seed(seed)

    # 1) Load data
    X, y = load_data(data_path)

    # 2) Build Dataset + split
    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val]) if n_val > 0 else (dataset, None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size) if val_set else None

    # 3) Instantiate submission model (no args) and wire optimizer to its internal net
    m = model()                     # <- required API
    net = m.net                     # <- nn.Sequential inside your class
    net.train()

    criterion = nn.BCELoss()        # because final layer in model.py uses Sigmoid
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 4) Train loop
    for epoch in range(1, epochs + 1):
        # -- train
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = net(xb)          # (N,1) with sigmoid already applied
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # -- validate
        if val_loader:
            net.eval()
            with torch.no_grad():
                val_loss_sum, correct, total = 0.0, 0, 0
                for xb, yb in val_loader:
                    pb = net(xb)
                    val_loss_sum += criterion(pb, yb).item() * xb.size(0)
                    preds = (pb.squeeze(1) > 0.5).float()
                    correct += (preds == yb.squeeze(1)).sum().item()
                    total += xb.size(0)
            val_loss = val_loss_sum / total
            val_acc = correct / total
            net.train()
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

    # 5) Save weights for the grader
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(net.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")

if __name__ == "__main__":
    main()
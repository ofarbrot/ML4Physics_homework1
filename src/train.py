import torch
import torch.nn as nn

def TrainingAlgorithm(model, dataloader, num_epochs, device="cpu"):
    '''
    Takes inn model, data_loader (training set and validation set) and num_epochs.
    '''
    losses = []

    # Choosing loss function and optimizer
    loss_func = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs): #num of full pass through the entire training dataset
        total_loss = 0.0 

        for X_batch, y_batch in dataloader:
            # Move data to device (GPU/CPU? figure out what to use, might be a problem for MAC (Synne))
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            # Forward pass
            y_pred = model(X_batch)
            loss = loss_func(y_pred, y_batch)

            # Backward pass
            '''
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            '''
            logits = model(X_batch)          # shape (N,)
            logits = logits.unsqueeze(1)     # -> (N,1)
            y_batch = y_batch.unsqueeze(1)
            loss = loss_func(logits, y_batch)
            

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

    # Save weights
    torch.save(model.state_dict(), "modells/model_weights.pth")

    return losses

def evaluate(model, val_loader, device="cpu"):
    model.eval()  # evaluation mode
    loss_fn = nn.BCELoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            preds = model.pred(X_batch).squeeze()

            loss = loss_fn(preds, y_batch)
            total_loss += loss.item()

            # thresholding for accuracy
            predicted_labels = (preds >= 0.5).long()
            correct += (predicted_labels == y_batch.long()).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy

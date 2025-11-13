import torch
import numpy as np
import sklearn
import torch.nn as nn
#from .model import model

def TrainingAlgoritm(model, dataloader, num_epochs, device="cpu"):
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
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            '''
            logits = model(X_batch)          # shape (N,)
            logits = logits.unsqueeze(1)     # -> (N,1)
            y_batch = y_batch.unsqueeze(1)
            loss = loss_func(logits, y_batch)
            '''

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

    # Save weights
    torch.save(model.state_dict(), "../models/model_weights.pth")

    return losses
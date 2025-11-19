import torch
import torch.nn as nn

def accuracy(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


def TrainingAlgorithm(model, train_loader, val_loader, num_epochs, device="cpu"):
    """
    Train the model on train_loader and evaluate on eval_loader.
    Return lists train_losses: avarage trainingloss per epoch, and 
    eval_losses: avarage validationloss per epoch
    """
    model.to(device)

    train_losses = []
    eval_losses = []
    eval_accuracies = []

    # Loss-function og optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):

        # -------- TRAINING --------
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            # Forward pass
            logits = model(X_batch)
            loss = loss_func(logits, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        total_eval_loss = 0.0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float()

                logits_val = model(X_val)
                loss_val = loss_func(logits_val, y_val)

                total_eval_loss += loss_val.item()

                all_logits.append(logits_val)
                all_targets.append(y_val)

        avg_eval_loss = total_eval_loss / len(val_loader)
        eval_losses.append(avg_eval_loss)

        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        val_acc = accuracy(logits_cat, targets_cat)
        eval_accuracies.append(val_acc)

        '''print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_eval_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")'''

    # -------- SAVE WEIGHTS --------
    torch.save(model.state_dict(), "src/model_weights.pth")

    return train_losses, eval_losses, eval_accuracies
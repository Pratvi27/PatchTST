"""Training, validation, and evaluation loop for PatchTST."""

import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model, device='cpu', best_model_path='./best_model.pt'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.best_model_path = best_model_path
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'test_loss': None,
        }

    def train_epoch(self, train_loader, optimizer, accumulation_steps=1):
        self.model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        i = -1
        for i, (x_batch, y_batch) in pbar:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.permute(0, 2, 1).to(self.device)

            out = self.model(x_batch)
            loss = self.criterion(out, y_batch) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        if i >= 0 and (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss / max(len(train_loader), 1)

    def validate(self, valid_loader):
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.permute(0, 2, 1).to(self.device)
                out = self.model(x_batch)
                valid_loss += self.criterion(out, y_batch).item()

        return valid_loss / len(valid_loader)

    def train(self, train_loader, valid_loader, epochs=10, lr=1e-4,
              accumulation_steps=1, patience=5):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_valid_loss = float('inf')
        patience_counter = 0

        os.makedirs(os.path.dirname(self.best_model_path) or '.', exist_ok=True)

        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Accumulation steps: {accumulation_steps}")
        print(f"  Device: {self.device}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Best model checkpoint: {self.best_model_path}")

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")

            train_loss = self.train_epoch(train_loader, optimizer, accumulation_steps)
            valid_loss = self.validate(valid_loader)
            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  [OK] New best model saved (Valid Loss: {best_valid_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"\n{'='*60}")
        print(f"Training completed. Best validation loss: {best_valid_loss:.4f}")
        print(f"{'='*60}")

        return self.history

    def evaluate(self, test_loader, return_predictions=False):
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []

        print("\nEvaluating on test set...")
        with torch.no_grad():
            for x_batch, y_batch in tqdm(test_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.permute(0, 2, 1).to(self.device)
                out = self.model(x_batch)
                test_loss += self.criterion(out, y_batch).item()

                if return_predictions:
                    all_predictions.append(out.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())

        test_loss = test_loss / len(test_loader)
        self.history['test_loss'] = test_loss

        print(f"\n{'='*60}")
        print(f"Test Loss (MSE): {test_loss:.4f}")
        print(f"Test RMSE: {np.sqrt(test_loss):.4f}")
        print(f"{'='*60}")

        if return_predictions:
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            return test_loss, predictions, targets

        return test_loss

    def get_history(self):
        return self.history

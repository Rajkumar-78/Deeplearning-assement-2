import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare and save a CSV so CsvDataset reads an actual CSV file.
data = load_breast_cancer()
X_train_all, X_test, y_train_all, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.2, random_state=SEED, stratify=y_train_all
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

columns = [f"feature_{i}" for i in range(X_train.shape[1])] + ["label"]
pd.DataFrame(np.column_stack([X_train, y_train]), columns=columns).to_csv("train.csv", index=False)
pd.DataFrame(np.column_stack([X_val, y_val]), columns=columns).to_csv("validation.csv", index=False)
pd.DataFrame(np.column_stack([X_test, y_test]), columns=columns).to_csv("test.csv", index=False)

class CsvDataset(Dataset):
    def __init__(self, csv_path):
        frame = pd.read_csv(csv_path)
        self.X = torch.tensor(frame.drop(columns=["label"]).to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(CsvDataset("train.csv"), batch_size=32, shuffle=True)
val_loader = DataLoader(CsvDataset("validation.csv"), batch_size=32, shuffle=False)
test_loader = DataLoader(CsvDataset("test.csv"), batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X_train.shape[1]).to(device)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {param_count:,}")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total

best_val = -1.0
best_state = None
patience, stale = 7, 0
train_losses = []
for epoch in range(1, 41):
    model.train()
    running_loss, seen = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * yb.size(0)
        seen += yb.size(0)
    train_loss = running_loss / seen
    val_acc = evaluate(model, val_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_accuracy={val_acc:.4f}")
    if val_acc > best_val:
        best_val, best_state, best_epoch, stale = val_acc, copy.deepcopy(model.state_dict()), epoch, 0
    else:
        stale += 1
        if stale >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "mlp_state_dict.pt")
test_acc = evaluate(model, test_loader)
print(f"Best val accuracy {best_val:.4f} at epoch {best_epoch}")
print(f"Test accuracy: {test_acc:.4f}")
reloaded = MLP(X_train.shape[1]).to(device)
reloaded.load_state_dict(torch.load("mlp_state_dict.pt", map_location=device, weights_only=True))
reloaded.eval()
reload_acc = evaluate(reloaded, test_loader)
print(f"Reloaded model accuracy: {reload_acc:.4f}")
print("Reload accuracy identical:", abs(reload_acc - test_acc) < 1e-12)
baseline = LogisticRegression(max_iter=2000, random_state=SEED)
baseline.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Scikit-learn LogisticRegression test accuracy: {lr_acc:.4f}")

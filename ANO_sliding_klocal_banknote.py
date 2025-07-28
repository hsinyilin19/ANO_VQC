import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
import pennylane as qml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_H', type=float, default=1e-1)
parser.add_argument('--n-qubits', type=int, default=4)
parser.add_argument('--n-local', type=int, default=1)
parser.add_argument('--vqc-depth', type=int, default=4)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()



def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """ Layer of CNOTs followed by another shifted layer of CNOT."""
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: epoch=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  epoch=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


# Define actual circuit architecture
def quantum_net(X, θ, H):
    """ The variational quantum circuit. """

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(args.n_qubits)

    # Embed features in the quantum node
    RY_layer(X)

    # Sequence of trainable variational layers
    for k in range(args.vqc_depth):
        entangling_layer(args.n_qubits)
        RY_layer(θ[k])

    # Compute Expectation values (for multi-class prediction) using n_local Hermitians
    exp_vals = [qml.expval(qml.Hermitian(H[q], wires=(np.arange(q, q + args.n_local) % args.n_qubits).tolist())) for q in range(args.n_qubits)]
    return exp_vals


def create_Hermitian(N, A, B, D):
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()  # fill diagonal
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()  # fill off-diagonal
        count += i
    H = h.clone() + h.clone().conj().T
    return H


class ANO_VQC_Model(nn.Module):
    '''VQC with adaptive nonlocal observables'''

    def __init__(self, img_size):
        super(ANO_VQC_Model, self).__init__()
        self.θ = nn.Parameter(0.01 * torch.randn(args.vqc_depth, args.n_qubits))  # VQC rotation params

        self.dev = qml.device("default.qubit", wires=args.n_qubits)  # Can use different simulation backend or quantum computers.
        self.VQC = qml.QNode(quantum_net, self.dev, interface = "torch")

        self.N = 2 ** args.n_local
        self.A = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(args.n_qubits)])  # off-diagonal entries (real part)
        self.B = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(args.n_qubits)])  # off-diagonal entries (imag part)
        self.D = nn.ParameterList([nn.Parameter(torch.empty(self.N)) for _ in range(args.n_qubits)])                        # diagonal elements (all real)

        for q in range(args.n_qubits):
            nn.init.normal_(self.A[q], std=2.)
            nn.init.normal_(self.B[q], std=2.)
            nn.init.normal_(self.D[q], std=2.)

    def forward(self, X):
        z1 = X.reshape(-1,args.n_qubits)

        # create observable H here
        self.H = [create_Hermitian(self.N, self.A[q], self.B[q], self.D[q]) for q in range(args.n_qubits)]
        q_out = torch.stack([torch.stack(self.VQC(z, self.θ, self.H)).float() for z in z1])
        return q_out[:, :2]

# Custom Dataset class
class BanknoteDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load the dataset
data = pd.read_csv("./banknote_authentication.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1].values  # First 4 columns (features)
#drop column name
X = X[1:].astype(float)

y = data.iloc[:, -1].values   # Last column (class labels)
y = y[1:].astype(float)


# Count occurrences of each class
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Class distribution before balancing:", class_counts)

# Determine the minimum class count for balancing
min_count = min(class_counts.values())

# Find indices of each class
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]

# Randomly sample from the majority class to match the minority class count
np.random.seed(42)
class_0_sampled = np.random.choice(class_0_indices, min_count, replace=False)
class_1_sampled = np.random.choice(class_1_indices, min_count, replace=False)

# Combine the balanced indices
balanced_indices = np.concatenate((class_0_sampled, class_1_sampled))
np.random.shuffle(balanced_indices)

# Extract the balanced dataset
X_balanced = X[balanced_indices]
y_balanced = y[balanced_indices]

# Verify the new class distribution
unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
class_counts_balanced = dict(zip(unique_balanced, counts_balanced))
print("Class distribution after balancing:", class_counts_balanced)

# Standardize the features
scaler = MinMaxScaler()

X_balanced = scaler.fit_transform(X_balanced)
X_balanced = X_balanced * (2 * np.pi) - np.pi


# Split into train and test sets
X_subset, X_test, y_subset, y_test = train_test_split(X_balanced, y_balanced, test_size=0.1, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_subset, y_subset, test_size=0.1, random_state=42)

# Create PyTorch datasets
train_dataset = BanknoteDataset(X_train, y_train)
valid_dataset = BanknoteDataset(X_valid, y_valid)
test_dataset = BanknoteDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check the dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validate samples: {len(valid_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")


img_size = train_dataset[0][0].shape
model = ANO_VQC_Model(img_size=img_size).to(DEVICE)


# Function to calculate trainable parameters
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

total_params = count_parameters(model)

# Calculate the number of trainable parameters in self.H
n_local = args.n_local
n_qubits = args.n_qubits
N = 2 ** n_local  # Dimension of each Hermitian matrix

# Split parameters into two groups
H_params = []
VQC_params = []
for name, param in model.named_parameters():
    if 'A' in name or 'B' in name or 'D' in name:
        H_params.append(param)  # Hermitian parameters
    else:
        VQC_params.append(param)

# initialize optimizer
H_optimizer = torch.optim.Adam(H_params, lr=args.lr_H)
optimizer = torch.optim.Adam(VQC_params, lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    print('*' * 30)
    print(f'Epoch {epoch + 1}'.center(30))
    print('*' * 30)

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    N = 0  # total samples

    for X, y in tqdm(train_loader):
        # send to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        N += len(y)  # accumulate batch sample size

        # set the gradient to zero
        optimizer.zero_grad()
        H_optimizer.zero_grad()

        # compute the model and the loss
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        batch_acc = sum(torch.argmax(logits, axis=1) == y) / len(y)
        total_acc += sum(torch.argmax(logits, axis=1) == y)
        loss.backward()
        optimizer.step()
        H_optimizer.step()

        print(f'[batch] train loss: {loss.item():.4f}, acc: {100 * batch_acc:.2f}%')

    train_epoch_acc = total_acc / N

    print(f'Epoch training loss {total_loss / N:.4f}, Train acc: {100 * train_epoch_acc:.2f}%')

    # validation
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    N = 0
    for X, y in tqdm(valid_loader):
        # send to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        N += len(y)  # accumulate batch sample size

        # compute the model and the loss
        with torch.no_grad():
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            total_acc += sum(torch.argmax(logits, axis=1) == y)

    valid_acc = total_acc / N
    print(f'Valid loss: {total_loss / N:.4f}, Valid acc: {100 * valid_acc:.2f}%')

    # Test model
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    N = 0
    for X, y in tqdm(test_loader):
        # send to device
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        N += len(y)  # accumulate batch sample size

        # compute the model and the loss
        with torch.no_grad():
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            total_acc += sum(torch.argmax(logits, axis=1) == y)

    test_acc = total_acc / N
    print(f'Test loss: {total_loss / N:.4f}, Test acc: {100 * test_acc:.2f}%')





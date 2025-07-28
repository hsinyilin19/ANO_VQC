import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pennylane as qml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import itertools


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_H', type=float, default=1e-1)
parser.add_argument('--n-qubits', type=int, default=16)
parser.add_argument('--n-local', type=int, default=2)
parser.add_argument('--vqc-depth', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=30)
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
# def quantum_net(X, θ, H):
#     """ The variational quantum circuit. """
#     # Start from state |+> , unbiased w.r.t. |0> and |1>
#     H_layer(args.n_qubits)
#
#     # Embed features in the quantum node
#     RY_layer(X)
#
#     # Sequence of trainable variational layers
#     for k in range(args.vqc_depth):
#         entangling_layer(args.n_qubits)
#         RY_layer(θ[k])
#
#     # Compute Expectation values (for multi-class prediction) using n_local Hermitians
#     exp_vals = [qml.expval(qml.Hermitian(H[q], wires=(np.arange(q, q + args.n_local) % args.n_qubits).tolist())) for q in range(args.n_qubits)]
#     return exp_vals

# Define actual circuit architecture
def quantum_net_any_comb_H(X, H):
    """ The variational quantum circuit. """

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(args.n_qubits)

    # Embed features in the quantum node
    RY_layer(X)

    entangling_layer(args.n_qubits)


    # Generate all possible combinations of two wires
    wire_combinations = list(itertools.combinations(range(args.n_qubits), args.n_local))

    # Compute the expectation values for each combination
    exp_vals = [
        qml.expval(qml.Hermitian(H[q], wires=list(combination)))
        for q, combination in enumerate(wire_combinations)
    ]
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
    '''Simple CNN model with Grad-CAM'''

    def __init__(self, img_size):
        super(ANO_VQC_Model, self).__init__()
        # self.θ = nn.Parameter(0.01 * torch.randn(args.vqc_depth, args.n_qubits))  # VQC rotation params
        wire_combinations = list(itertools.combinations(range(args.n_qubits), args.n_local))
        num_H = len(wire_combinations)

        self.linear = nn.Linear(num_H, 10)

        self.dev = qml.device("default.qubit", wires=args.n_qubits)  # Can use different simulation backend or quantum computers.
        self.VQC = qml.QNode(quantum_net_any_comb_H, self.dev, interface = "torch")

        self.N = 2 ** args.n_local

        self.A = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(num_H)])   # off-diagonal entries (real part)
        self.B = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in
                                   range(num_H)])  # off-diagonal entries (imag part)
        self.D = nn.ParameterList(
            [nn.Parameter(torch.empty(self.N)) for _ in range(num_H)])  # diagonal elements (all real)

        for q in range(args.n_qubits):
            nn.init.normal_(self.A[q], std=2.)
            nn.init.normal_(self.B[q], std=2.)
            nn.init.normal_(self.D[q], std=2.)

    def forward(self, X):
        z1 = X.reshape(-1,args.n_qubits)

        wire_combinations = list(itertools.combinations(range(args.n_qubits), args.n_local))

        # create observable H here
        self.H = [create_Hermitian(self.N, self.A[q], self.B[q], self.D[q]) for q in range(len(wire_combinations))]
        q_out = torch.stack([torch.stack(self.VQC(z, self.H)).float() for z in z1])
        output=self.linear(q_out)
        return output



# Define a custom transformation
class NormalizeToPiTransform:
    def __call__(self, x):
        """
        Transform values from range [0, 1] to [-pi, pi].
        Assumes input x is already normalized to [0, 1].
        """
        return x * (2 * np.pi) - np.pi

# Compose transformations
transform = transforms.Compose([
    transforms.Resize((4, 4)),
    transforms.ToTensor(),  # Convert image to tensor with range [0, 1]
    NormalizeToPiTransform(),  # Scale to [-pi, pi]
])

# Download the MNIST dataset
train_dataset = datasets.MNIST(
    root="mnist_data",  # Directory to save the dataset
    train=True,         # Download the training set
    transform=transform,  # Apply the transform
    download=True       # Download if not already downloaded
)

test_dataset = datasets.MNIST(
    root="mnist_data",
    train=False,        # Download the test set
    transform=transform,
    download=True
)


# Parameters
num_classes = 10  # MNIST has 10 classes: digits 0-9
samples_per_class_train = 1000  # Number of samples per class in the subset
samples_per_class_test = 100

# Initialize a dictionary to store indices for each class
class_indices = {i: [] for i in range(num_classes)}

# Populate the dictionary with indices
for idx, (_, label) in enumerate(train_dataset):
    if len(class_indices[label]) < samples_per_class_train:  # Check if class is filled
        class_indices[label].append(idx)

# Combine all indices from each class
balanced_indices = [idx for indices in class_indices.values() for idx in indices]

# Create a subset using the balanced indices
balanced_train_subset = Subset(train_dataset, balanced_indices)

# Initialize a dictionary to store indices for each class
class_indices = {i: [] for i in range(num_classes)}

# Populate the dictionary with indices
for idx, (_, label) in enumerate(test_dataset):
    if len(class_indices[label]) < samples_per_class_test:  # Check if class is filled
        class_indices[label].append(idx)

# Combine all indices from each class
balanced_indices = [idx for indices in class_indices.values() for idx in indices]

# Create a subset using the balanced indices
balanced_test_subset = Subset(test_dataset, balanced_indices)

train_dataset_from_subset, valid_dataset_from_subset = train_test_split(balanced_train_subset, test_size=0.1, random_state=999, shuffle=True)


# Create DataLoader for training and testing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset_from_subset, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset_from_subset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=balanced_test_subset, batch_size=args.batch_size, shuffle=False)

# Check the dataset
print(f"Number of training samples: {len(train_dataset_from_subset)}")
print(f"Number of validate samples: {len(valid_dataset_from_subset)}")
print(f"Number of testing samples: {len(balanced_test_subset)}")

# Example: Display one batch of images
data_iter = iter(train_loader)
images, labels = next(data_iter)


img_size = train_dataset[0][0].shape
model = ANO_VQC_Model(img_size=img_size).to(DEVICE)


# Calculate the number of trainable parameters in self.H
n_local = args.n_local
n_qubits = args.n_qubits
N = 2 ** n_local  # Dimension of each Hermitian matrix

# Parameters per Hermitian matrix
params_per_matrix = (N * (N - 1)) // 2 * 2 + N  # Off-diagonal (real + imag) + diagonal

wire_combinations = list(itertools.combinations(range(args.n_qubits), args.n_local))
num_H = len(wire_combinations)


# Split parameters into two groups
H_params = []
Linear_params = []
for name, param in model.named_parameters():
    if 'A' in name or 'B' in name or 'D' in name:
        H_params.append(param)  # Hermitian parameters
    else:
        Linear_params.append(param)

# initialize optimizer
H_optimizer = torch.optim.Adam(H_params, lr=args.lr_H)
optimizer = torch.optim.Adam(Linear_params, lr=args.lr)
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

        # print(f'H: {model.H[0]}')

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

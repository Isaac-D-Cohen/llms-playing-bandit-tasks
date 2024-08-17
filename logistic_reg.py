import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

    
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

X = np.load(f"pc_values.npy")
# y = np.concatenate((np.zeros(1190), np.ones(1190), np.full(1190, 2, dtype=int)))
y = np.concatenate((np.zeros(400), np.ones(400)))
# print(x.shape, y.shape)
num_layers = X.shape[0]

input_dim = X.shape[2]
output_dim = 2 # greedy and antigreedy
model = LogisticRegressionModel(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

num_epochs = 100
accuracies = []

for i in range(num_layers):
    #print(f"layer {i} stuff")
    x = X[i]
    #print(x.shape, y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    for epoch in range(num_epochs):
        model.train()

        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (epoch+1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
        accuracies.append(accuracy * 100) 
        #print(f'Accuracy: {accuracy * 100:.2f}%\n')
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_layers + 1), accuracies, marker='o')
plt.xlabel('Layer Number')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Layer Number')
plt.grid(True)
plt.show()

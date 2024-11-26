import torch
import os
import pickle
import numpy as np
from training_keras import NeuralNet, hidden_size, X_test, Y_test
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the scaler
pickle_dir = 'pickles'
with open(os.path.join(pickle_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# sacling the test values 
X_test_scaled = scaler.fit_transform(X_test) 
Y_test_scaled = scaler.fit_transform(Y_test)

# Ensure data is converted to tensors
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
Y_test_scaled = torch.tensor(Y_test_scaled, dtype=torch.float32).to(device)

# inputting the input_size and hidden_size variables
input_size = X_test.shape[1]

# Load the saved model
model = NeuralNet(input_size, hidden_size).to(device)  # Ensure you initialize with the correct input_size and hidden_size
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))  # Replace 'model_weights.pth' with your saved model weights file
model.eval()  # Set model to evaluation mode

# Make predictions
with torch.no_grad():
    y_pred_scaled = model(X_test_scaled).cpu().numpy()  # Get scaled predictions
    Y_test_scaled = Y_test_scaled.cpu().numpy()

y_pred = scaler.inverse_transform(y_pred_scaled)
Y_test = scaler.inverse_transform(Y_test_scaled)

# Select 20 samples for visualization
num_samples = 20
indices = np.arange(num_samples)
y_pred_subset = y_pred[:num_samples].flatten()
Y_test_subset = Y_test[:num_samples].flatten()

# Create a bar chart for side-by-side comparison
bar_width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting true values and predictions
ax.bar(indices - bar_width / 2, Y_test_subset, bar_width, label='True Values', color='blue')
ax.bar(indices + bar_width / 2, y_pred_subset, bar_width, label='Predicted Values', color='red')

# Add labels and legend
ax.set_title('Comparison of True Values vs Predicted Values (20 Samples)')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Estimated Salary')
ax.set_xticks(indices)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
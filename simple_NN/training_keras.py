import os
import torch
import pickle
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# tensorboard 
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(log_dir)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initializing the encoders and scaler
label_encoder_gender = LabelEncoder()
label_ohe_geo = OneHotEncoder()
scaler = StandardScaler()

# Hyper-parameters 
hidden_size = 100
batch_size = 4
learning_rate = 0.001

# data loading and preprocessing
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# encoding the gender col
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# encoding the geo column
geo_encoder = label_ohe_geo.fit_transform(data[['Geography']])
cols = label_ohe_geo.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoder.toarray(), columns=cols)
data_encoded = pd.concat([data.drop(['Geography'], axis=1), geo_encoded_df], axis=1)

# defining the input features and output 
X = data_encoded.drop(['EstimatedSalary'], axis=1)
Y = data_encoded['EstimatedSalary'].values.reshape(-1, 1)

# splitting the data to train and test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# scaling the data and converting to tesnsors 
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
Y_train = torch.tensor(scaler.fit_transform(Y_train), dtype=torch.float32)  

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# after encoding the values we save encoders as pickle file for the usae of streamlit
pickle_dir = 'pickles'

# ensuring that the directory exists
os.makedirs(pickle_dir, exist_ok=True)  

pickle_files = {
    'label_encoder_gender.pkl': label_encoder_gender,
    'geo_ohencoder.pkl': label_ohe_geo,
    'scaler.pkl': scaler
}

# Save each object to its respective file
for filename, obj in pickle_files.items():
    with open(os.path.join(pickle_dir, filename), 'wb') as file:
        pickle.dump(obj, file)

# using keras library nn, but this can be rewritten using tensorflow
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, 1)  # Output layer (salary prediction)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)  # No activation at the end for regression
        return out

input_size = X_train.shape[1]

# initializing the defined model
model = NeuralNet(input_size, hidden_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# train the model
num_epochs = 50
loss_threshold = 0.1

with open(os.path.join(pickle_dir, 'X_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)

with open(os.path.join(pickle_dir, 'Y_test.pkl'), 'wb') as f:
    pickle.dump(Y_test, f)

# Close TensorBoard writer
writer.close()

if __name__ == "__main__":
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            # Forward pass
            y_predicted = model(batch_X)
            loss = criterion(y_predicted, batch_Y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        if loss.item() < loss_threshold:
            print(f"Loss is below {loss_threshold}, saving the model...")
            torch.save(model.state_dict(), 'model_weights.pth')  # Saves the model weights
            
    print("Training logic executed.")
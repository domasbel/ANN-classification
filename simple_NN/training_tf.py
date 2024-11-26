import os
import sys
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard

# tensorboard
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

# logs directory
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# initializing the encoders and scaler
label_encoder_gender = LabelEncoder()
label_ohe_geo = OneHotEncoder()
scaler = StandardScaler()

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

# scaling the data for normalization purposes
X_train_scaled = scaler.fit_transform(X_train)
Y_train_scaled = scaler.fit_transform(Y_train)

# validation set aswell
X_test_scaled = scaler.fit_transform(X_test)
Y_test_scaled = scaler.fit_transform(Y_test)

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

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# unit here we have data processing and preparation, which is the same for both libraries
        # only difference, here we dont need to transform data to tensors and thus data_loader can be avoided aswell
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# defining model inputs 
input_size = (X_train.shape[1], )
hidden_size = 64 # for easier readability i insert 0 into the first place

# defining the model
model = Sequential([
    Dense(hidden_size, activation='relu', input_shape=input_size), # this is the HL1, from input layer
    Dense(1) # output layer, since its regression we dont need any activation function
])

# model.summary()

# setting up the training loss and optimizer function

opt = tf.keras.optimizers.Adagrad(learning_rate=0.001)
# other viable options are SGD, Adagrad, AdamW 

loss = tf.keras.losses.MeanSquaredError() 
# other viable options are MeanAbsoluteError, Huber, LogCosh or a custom (?)

# defining the training model parameters
model.compile(
    optimizer=opt,
    loss=loss,
    metrics = ['mse'] # here a custom metric can be used if needed, but we will only monitor the one we have used for the model training as loss 
)

# for a more accurate and friendly training process we will add earlystop to not waste time if selected hyperparameters are invalid
early_stopping_callback = EarlyStopping(
    monitor='val_mse',
    patience=30,
    restore_best_weights=True,
    min_delta=0.001, # this sets so that minimal improvement before patience reset is 0.001
)

# defining a callback class that saves the model when we reach desired threshold
class SaveModelOnMSE(Callback):
    def __init__(self, mse_threshold=0.1, model_save_name='best_model', save_dir='models/'):
        super(SaveModelOnMSE, self).__init__()
        self.mse_threshold = mse_threshold
        self.model_save_dir = save_dir
        self.model_save_name = model_save_name
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Check the validation MSE (or loss) at the end of each epoch
        val_loss = logs.get('val_loss')
        if val_loss and val_loss < self.mse_threshold:
            print(f"\nEpoch {epoch+1}: Validation loss {val_loss:.4f} is below the threshold {self.mse_threshold}. Saving the model.")
            model_save_path = os.path.join(self.model_save_dir, self.model_save_name + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.h5')
            self.model.save(model_save_path)
            print(f"Model saved at {model_save_path}")

# execution the model training 
# defining so that it would be executed only the the file is executed, to avoid executions during the testing period, when taking data
if __name__ == '__main__':

    tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    save_model_callback = SaveModelOnMSE(mse_threshold=0.1, model_save_name='best_model')

    model.fit(
        X_train_scaled,
        Y_train_scaled,
        validation_data = (X_test_scaled, Y_test_scaled),
        epochs = 50,
        batch_size = 16,
        callbacks = [tensorflow_callback, early_stopping_callback, save_model_callback]
    )

    print("Training logic executed.")
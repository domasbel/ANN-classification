# ANN-Regression

Customer Income Prediction Model

This project implements a regression model designed to predict the estimated income of a customer based on 12 different features. The primary goal of this model is to demonstrate the ability to preprocess data, design a neural network, and apply hyperparameter tuning for optimal performance.

1. Key Features and Techniques Used:
- Data Preprocessing - Label Encoding and One-Hot Encoding for handling categorical variables, ensuring that all inputs are numerical and suitable for model training.
- Data Scaling using a scaler to normalize continuous variables, improving the convergence and performance of the model.
  
2. Modeling:
- A simple neural network (NN) is created using TensorFlow and Keras, where separate implementations of the model are provided in both libraries to showcase proficiency in each.
- The model predicts the target variable, which is the estimated income of a customer.
  
3. Hyperparameter Tuning:
- A Grid Search approach is applied to explore various hyperparameter combinations (such as the number of layers, number of neurons, and epochs) and identify the best model architecture based on Mean Squared Error (MSE), the chosen metric for regression tasks.
- K-Fold Cross Validation is used during the grid search to ensure the model's robustness and prevent overfitting.
  
4. Tools and Libraries:
- TensorFlow & Keras: For creating and training the neural network.
- Scikit-learn: For data preprocessing (encoding, scaling) and model evaluation.
- GridSearchCV: For optimizing the hyperparameters of the model.


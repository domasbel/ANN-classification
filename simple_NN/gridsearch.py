from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from training_tf import input_size, X_train_scaled, Y_train_scaled

param_grid = {
    'neurons': [64, 128, 256],
    'layers': [2, 3, 4],
    'epochs': [50, 75, 100]
}

def create_model(neurons=32, layers=1):
    model = Sequential()
    model.add(Input(shape = (X_train_scaled.shape[1], )))
    model.add(Dense(neurons, activation='relu'))

    for _ in range(layers-1):
        model.add(Dense(neurons, activation='relu'))
    
    model.add(Dense(1))
    model.compile(
        optimizer = 'adam',
        loss = 'mse',
        metrics = ['accuracy']
    )

    return model

model =  KerasRegressor(
    layers = 1,
    neurons = 32,
    model = create_model, # here we define what is our grid search for the model architecture
    verbose = 1
)

grid = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        n_jobs = -1,
        cv = 3)

grid_result = grid.fit(X_train_scaled, Y_train_scaled)

print('Best: %f, using %s' % (grid_result.best_score_, grid_result.best_params_))
# first result -  Best: -0.040155, using {'epochs': 50, 'layers': 1, 'neurons': 64}
# after increasing the count of layers we got - Best: -0.177289, using {'epochs': 50, 'layers': 2, 'neurons': 64}

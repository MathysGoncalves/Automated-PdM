import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import Input, Dropout, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping, ModelCheckpoint

import joblib


def minmaxscaler(train):
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaler_path = "../models/scaler_data.pkl"
    joblib.dump(scaler, scaler_path)
    return scaler_path

def apply_scaler(data):
    scaler = joblib.load("../models/scaler_data.pkl")
    return scaler.transform(data)

def create_autoencoder(input_dim, num_hidden_layers, hidden_layer_sizes, dropout_rate, learning_rate):
    inputs = Input(shape=(input_dim,))
    encoded = inputs
    
    for layer_size in range(num_hidden_layers):
        encoded = Dense(int((hidden_layer_sizes/(layer_size+1))), activation="selu", kernel_initializer="lecun_normal")(encoded)    # selu + lecun = avoid exploding gradient
        encoded = Dropout(rate=dropout_rate)(encoded)
    decoded = encoded

    for layer_size in range(num_hidden_layers):
        decoded = Dense(int((hidden_layer_sizes/(layer_size+1))), activation="selu", kernel_initializer="lecun_normal")(decoded)
        encoded = Dropout(rate=dropout_rate)(encoded)
    decoded = Dense(input_dim, activation='relu')(decoded)

    autoencoder = Model(inputs, decoded)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    #autoencoder.summary()
    return autoencoder

def learning_curves(autoencoder_search):
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(autoencoder_search.best_estimator_.model.history.history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(autoencoder_search.best_estimator_.model.history.history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()

def fit(train_scaled):
    # Define the search space and search for the best hyperparameters using RandomizedSearchCV
    params = {
        'input_dim': [4, None],
        'num_hidden_layers': range(1, 5),
        'hidden_layer_sizes': [8, 16, 32, 64],
        'learning_rate': [10 ** i for i in range(-4, -1)],
        'dropout_rate': [0, 0.1, 0.2, 0.4]
    }

    autoencoder_model = KerasRegressor(build_fn=create_autoencoder)

    # create the randomized search object
    autoencoder_search = RandomizedSearchCV(estimator=autoencoder_model,
                                            param_distributions=params,
                                            cv=5,
                                            random_state=42,
                                            n_jobs=3)

    early_stopping = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    checkpoint = ModelCheckpoint(filepath='../models/autoencoder.h5',
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min')

    try:
        os.remove("../models/autoencoder.h5")
    except:
        print("No model saved yet")

    # fit the randomized search object to the training data
    autoencoder_search.fit(train_scaled,train_scaled,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.1,
                        shuffle=True,
                        callbacks=[early_stopping, checkpoint],
                        verbose=0)

    print(autoencoder_search.best_params_)
    return autoencoder_search


def cluster_data(autoencoder_search, train_scaled, test_scaled, test):
    reconstructions = autoencoder_search.predict(train_scaled)
    train_mae_loss = np.mean(np.abs(reconstructions - train_scaled), axis=1)

    reconstructions = autoencoder_search.predict(test_scaled)
    test_mae_loss = np.mean(np.abs(reconstructions - test_scaled), axis=1)

    #threshold = np.max(train_mae_loss)
    threshold = np.mean(train_mae_loss) + 2*np.std(train_mae_loss)
    print("Global Reconstruction error threshold: ", threshold)

    test['Loss'] = test_mae_loss
    test.loc[test['Loss'] <= threshold, 'Anomaly'] = "Normal"
    test.loc[test['Loss'] >= threshold, 'Anomaly'] = "Anomaly"
    test['Threshold'] = threshold

    test.to_csv('../data/final/test_Bearing_Nasa_AD.csv')
    print(test.head())
    return threshold

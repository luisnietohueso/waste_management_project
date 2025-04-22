"""
Author: Luis Nieto Hueso
Date: 28/03/2025
Description:
This script implements a complete deep learning pipeline for predicting waste bin filling levels using both traditional
LSTM-based Recurrent Neural Networks (RNNs) and hybrid CNN+LSTM models. It also includes a hyperparameter tuning phase
using Keras Tuner to optimise model performance.

The script begins by importing essential libraries:
- `pandas`, `numpy`, and `matplotlib.pyplot` for data manipulation and visualisation
- `sklearn` modules for preprocessing and splitting the data
- `tensorflow.keras` components for model building, training, and optimisation
- `hmmlearn`, `keras_tuner`, and `statsmodels` for additional modelling and statistical tools

Feature Engineering:
The `create_optimized_features` function transforms the raw fill-level data by scaling values using MinMaxScaler,
encoding categorical types (if necessary), and generating lag features based on autocorrelation insights.
It also calculates rolling statistics (mean, median, std), first- and second-order differences, and interaction features.
Outliers are removed using the IQR method and NaN values are dropped to ensure clean data input.

Data Preparation for RNN:
The `prepare_rnn_data` function prepares the feature-engineered data for model training.
It splits the data into training and test sets, normalises both features and targets, and reshapes the data
into the 3D format required for CNN/LSTM networks.

Model Tuning:
The `build_model` function defines a tunable LSTM model architecture.
The `RandomSearch` tuner searches for the best combination of units, dropout rates, dense layers, and learning rate
based on validation loss. The best model is selected and trained on the prepared dataset.

CNN + LSTM Model:
The `build_cnn_lstm_model` function builds a hybrid deep learning architecture that first applies
1D convolution and max-pooling to extract local temporal patterns, followed by stacked LSTM layers
to capture long-term dependencies. Regularisation and dropout are used to reduce overfitting.

Learning Rate Warm-up Scheduler:
The `warmup_lr` function gradually increases the learning rate during the first few epochs to stabilise early training.

**Model Training Function:**
The `train_rnn` function defines a separate RNN training pipeline with early stopping and learning rate reduction callbacks.
It is useful when not using hyperparameter tuning.

Prediction:
The `make_predictions` and `predict_hybrid` functions perform predictions using the trained models
and convert them back to the original fill-level scale using the saved scaler.

Overall, this script enables the analysis and forecasting of bin filling trends using advanced deep learning models
tailored for time-series data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from statsmodels.graphics.tsaplots import plot_acf
from keras_tuner.tuners import RandomSearch



#  Step  Feature Engineering
def create_optimized_features(data):
    scaler = MinMaxScaler()

    # Convert 'filling_level' to numeric
    if data['filling_level'].dtype == object:
        label_encoder = LabelEncoder()
        data['filling_level'] = label_encoder.fit_transform(data['filling_level'])

    # Normalize filling levels
    data['filling_level_scaled'] = scaler.fit_transform(data[['filling_level']])

    # Plot Autocorrelation to determine optimal lags
    plot_acf(data['filling_level_scaled'], lags=30)
    plt.title("Autocorrelation of Filling Level")
    plt.show()

    optimal_lags = [1, 3, 5, 7, 10]

    for lag in optimal_lags:
        data[f'Lag_{lag}'] = data['filling_level_scaled'].shift(lag)

    # Rolling statistics
    data['Rolling_Mean_3'] = data['filling_level_scaled'].rolling(window=3).mean()
    data['Rolling_Median_5'] = data['filling_level_scaled'].rolling(window=5).median()
    data['Rolling_Std_5'] = data['filling_level_scaled'].rolling(window=5).std()
    data['Diff_1'] = data['filling_level_scaled'].diff(1)
    data['Diff_2'] = data['filling_level_scaled'].diff(2)

    # Interaction Features
    data['Rolling_Mean_3_x_Lag_1'] = data['Rolling_Mean_3'] * data['Lag_1']
    data['Rolling_Std_5_x_Diff_1'] = data['Rolling_Std_5'] * data['Diff_1']

    # Remove outliers using IQR method
    Q1 = data['filling_level_scaled'].quantile(0.25)
    Q3 = data['filling_level_scaled'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['filling_level_scaled'] >= (Q1 - 1.5 * IQR)) & (data['filling_level_scaled'] <= (Q3 + 1.5 * IQR))]

    # Drop NaN rows
    data.dropna(inplace=True)

    feature_columns = [f'Lag_{lag}' for lag in optimal_lags] + [
        'Rolling_Mean_3', 'Rolling_Median_5', 'Rolling_Std_5', 'Diff_1', 'Diff_2',
        'Rolling_Mean_3_x_Lag_1', 'Rolling_Std_5_x_Diff_1'
    ]

    return data, feature_columns, scaler


#  Step  Data Preparation for RNN
def prepare_rnn_data(data):
    data, feature_columns, scaler = create_optimized_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], data['filling_level'], test_size=0.2, random_state=42
    )

    # Normalize features
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # Normalize target variable
    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))

    # Reshape for CNN+LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, scaler


# 🚀 Load Data & Prepare RNN Data Before Hyperparameter Tuning
# 🚀 Load Data & Prepare RNN Data Before Hyperparameter Tuning
file_path = r"E:\final year university\waste_management_project\waste_management_project\cleaned_bins_dataset.csv"  # Use full path
try:
    data = pd.read_csv(file_path)
    print(" Dataset loaded successfully!")
except FileNotFoundError:
    print(f" ERROR: File not found at {file_path}")
    exit(1)  # Stop execution if file is missing

X_train, X_test, y_train, y_test, scaler = prepare_rnn_data(data)  # Ensure data is ready before tuning



#  Step : Hyperparameter Tuning
def build_model(hp):
    model = Sequential([
        LSTM(hp.Int('units', min_value=32, max_value=128, step=16), activation='tanh', return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(hp.Float('dropout', 0.2, 0.4, step=0.05)),
        LSTM(hp.Int('units_2', min_value=16, max_value=64, step=16), activation='tanh'),
        Dense(hp.Int('dense_units', min_value=16, max_value=64, step=16), activation='relu'),
        Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.05)),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=hp.Choice('lr', [0.0003, 0.0005, 0.001])), loss='mse')
    return model


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='tuner_results',
    project_name='rnn_tuning'
)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))  # Now X_train is defined

#  Step  Train Optimized RNN
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

best_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=64)


#  Step  CNN + LSTM Hybrid Model
def build_cnn_lstm_model(input_shape):
    """
    Builds a hybrid CNN + LSTM model for better pattern extraction.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='tanh'),
        Dense(16, activation='relu', kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)  # Gradient Clipping
    model.compile(optimizer=optimizer, loss=Huber(delta=1.0))

    return model


#  Step  Learning Rate Finder & Warm-up Scheduler
def warmup_lr(epoch, lr):
    return lr * (epoch + 1) / 10 if epoch < 10 else lr


lr_scheduler = LearningRateScheduler(warmup_lr)


def train_rnn(X_train, y_train, X_test, y_test, epochs=120, batch_size=64):
    """
    Builds and trains an optimized LSTM model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32, activation='tanh'),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr], verbose=1)

    return model

#  Step  Make Predictions
def make_predictions(model, X, scaler):
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions)

#  Step  Predict Using the Optimized RNN
def predict_hybrid(model, X, scaler):
    """
    Makes predictions using the trained RNN and scales them back to the original range.
    """
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions)


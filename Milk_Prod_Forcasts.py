

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


def run_precip_forecast_pipeline(prep_df):
    # ----------------------------- #
    # MONTE CARLO DROPOUT FUNCTION #
    # ----------------------------- #
    def monte_carlo_predictions(model, X_input, n_simulations=100):
        @tf.function
        def f_model_mc(X_input, training=True):
            return model(X_input, training=training)

        preds = np.array([f_model_mc(X_input, training=True).numpy() for _ in range(n_simulations)])
        return preds.mean(axis=0), preds.std(axis=0)

    # ----------------------------- #
    # PREPROCESSING FUNCTION       #
    # ----------------------------- #
    def preprocess_data(data, target_var_idx, seq_length=12):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length, :])
            y.append(data[i + seq_length, target_var_idx])
        return np.array(X), np.array(y)

    # ----------------------------- #
    # MODEL TRAINING FUNCTION      #
    # ----------------------------- #
    def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, epochs=70, batch_size=16):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(16),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, mode='min', restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        return model, history

    # ----------------------------- #
    # EVALUATION FUNCTION          #
    # ----------------------------- #
    def evaluate_model(model, X_test, y_test, scaler, target_var_idx, full_scaled_data):
        y_pred_mean, y_pred_std = monte_carlo_predictions(model, X_test)

        def rescale(pred, std_or_target):
            dummy = np.zeros((len(pred), full_scaled_data.shape[1]))
            dummy[:, target_var_idx] = std_or_target
            return scaler.inverse_transform(dummy)[:, target_var_idx]

        y_pred_mean_rescaled = rescale(y_pred_mean, y_pred_mean.flatten())
        y_pred_std_rescaled = rescale(y_pred_std, y_pred_std.flatten())
        y_test_rescaled = rescale(y_test, y_test)

        valid = ~np.isnan(y_test_rescaled) & ~np.isnan(y_pred_mean_rescaled)
        y_test_rescaled, y_pred_mean_rescaled, y_pred_std_rescaled = \
            y_test_rescaled[valid], y_pred_mean_rescaled[valid], y_pred_std_rescaled[valid]

        return y_test_rescaled, y_pred_mean_rescaled, y_pred_std_rescaled

    # ----------------------------- #
    # RESULT COMPILATION FUNCTION  #
    # ----------------------------- #
    def compile_forecast_df(data, y_true, y_pred, y_std, target_var_idx, seq_length):
        start_idx = len(data) - len(y_true) - seq_length
        test_indices = list(range(start_idx, len(data) - seq_length))

        test_months = data.iloc[test_indices]["month_num"].values
        test_years = data.iloc[test_indices]["year"].values
        test_seasons = data.iloc[test_indices]["Season_Index"].values
        test_gaps = data.iloc[test_indices]["months_gap"].values

        forecast_df = pd.DataFrame({
            "Month": test_months,
            "Year": test_years,
            "Season_Index": test_seasons,
            "Months Gap": test_gaps,
            "Actual Precipitation": y_true,
            "Forecasted Precipitation": y_pred,
            "Forecast Uncertainty (Std Dev)": y_std,
            "Lower Bound (95%)": y_pred - 1.96 * y_std,
            "Upper Bound (95%)": y_pred + 1.96 * y_std,
            "Error": y_true - y_pred,
            "Percent Error": ((y_true - y_pred) / y_true) * 100
        })

        return forecast_df

    # ----------------------------- #
    # PERFORMANCE METRICS FUNCTION #
    # ----------------------------- #
    def get_performance_metrics(y_true, y_pred, lower, upper):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ci_coverage = ((y_true >= lower) & (y_true <= upper)).mean() * 100

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "CI Coverage (%)": ci_coverage
        }

    # ----------------------------- #
    # MAIN RUN PIPELINE            #
    # ----------------------------- #
    def run_precip_forecast_pipeline(data: pd.DataFrame):
        data = data.sort_values(['year', 'month_num']).copy()
        data['month_year'] = data['year'] * 12 + data['month_num']
        data['months_gap'] = data['month_year'].diff().fillna(1).astype(int)
        data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month_num'].astype(str).str.zfill(2) + '-01')

        features = ["year", "month_num", "Season_Index", "precipitation", "months_gap"]
        data_model = data[features]

        # Scaling
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_model)

        target_var_idx = 3  # precipitation
        seq_length = 48
        X, y = preprocess_data(data_scaled, target_var_idx, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        # Train model
        model, history = train_lstm_model(X_train, y_train, X_test, y_test, input_shape=(seq_length, X.shape[2]))

        # Evaluate model
        y_true, y_pred, y_std = evaluate_model(model, X_test, y_test, scaler, target_var_idx, data_scaled)

        # Compile results
        forecast_df = compile_forecast_df(data, y_true, y_pred, y_std, target_var_idx, seq_length)
        metrics = get_performance_metrics(y_true, y_pred, forecast_df["Lower Bound (95%)"], forecast_df["Upper Bound (95%)"])

        return {
            "model": model,
            "scaler": scaler,
            "forecast_df": forecast_df,
            "metrics": metrics,
            "history": history
        }
    

    return {
        "model": model,
        "forecast_df": forecast_df,
        "metrics": metrics
}


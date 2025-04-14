import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_precip_forecast_pipeline(prep_df, seq_length=48, forecast_start_year=2016, 
                               forecast_start_month=1, n_future=109, n_simulations=100,
                               test_size=0.1, epochs=70, batch_size=16):
    """
    Run the precipitation forecasting pipeline and return results without plotting.
    
    Args:
        prep_df: DataFrame containing precipitation data with columns:
                 year, month_num, Season_Index, precipitation
        seq_length: Sequence length for LSTM model
        forecast_start_year: Year to start forecasting from
        forecast_start_month: Month to start forecasting from
        n_future: Number of months to forecast
        n_simulations: Number of Monte Carlo dropout simulations
        test_size: Proportion of data for testing
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with model results and forecasts
    """
    # Process data
    data = prep_df.copy()
    data = data.sort_values(['year', 'month_num'])
    data['month_year'] = data['year'] * 12 + data['month_num']
    data['months_gap'] = data['month_year'].diff().fillna(1).astype(int)
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + 
                                data['month_num'].astype(str).str.zfill(2) + '-01')
    
    # Feature extraction
    features = ["year", "month_num", "Season_Index", "precipitation", "months_gap"]
    data_for_model = data[features].copy()
    feature_indices = {'year': 0, 'month': 1, 'season': 2, 'precip': 3, 'gap': 4}
    
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_for_model)
    
    # Create sequences
    def create_sequences(data, target_idx, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len, :])
            y.append(data[i + seq_len, target_idx])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(data_scaled, feature_indices['precip'], seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Build and train model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, X.shape[2])),
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
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Define prediction function
    def monte_carlo_predictions(model, X_input, n_sim=100):
        @tf.function
        def f_model_mc(X_input, training=True):
            return model(X_input, training=training)
        
        preds = np.array([f_model_mc(X_input, training=True).numpy() for _ in range(n_sim)])
        return preds.mean(axis=0), preds.std(axis=0)
    
    # Test predictions
    y_pred_mean, y_pred_std = monte_carlo_predictions(model, X_test, n_simulations)
    
    # Rescale predictions
    def rescale_predictions(predictions, std_devs, y_true, target_idx, scaler, n_features):
        # Prepare arrays for inverse scaling
        dummy_pred = np.zeros((len(predictions), n_features))
        dummy_pred[:, target_idx] = predictions.flatten()
        predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, target_idx]
        
        dummy_std = np.zeros((len(std_devs), n_features))
        dummy_std[:, target_idx] = std_devs.flatten()
        std_devs_rescaled = scaler.inverse_transform(dummy_std)[:, target_idx]
        
        if y_true is not None:
            dummy_y = np.zeros((len(y_true), n_features))
            dummy_y[:, target_idx] = y_true
            y_true_rescaled = scaler.inverse_transform(dummy_y)[:, target_idx]
            return predictions_rescaled, std_devs_rescaled, y_true_rescaled
        
        return predictions_rescaled, std_devs_rescaled
    
    # Rescale test predictions
    y_pred_mean_rescaled, y_pred_std_rescaled, y_test_rescaled = rescale_predictions(
        y_pred_mean, y_pred_std, y_test, feature_indices['precip'], scaler, X.shape[2]
    )
    
    # Calculate test metrics
    test_metrics = {}
    valid_indices = ~np.isnan(y_test_rescaled) & ~np.isnan(y_pred_mean_rescaled)
    
    if sum(valid_indices) > 0:
        y_test_valid = y_test_rescaled[valid_indices]
        y_pred_valid = y_pred_mean_rescaled[valid_indices]
        y_std_valid = y_pred_std_rescaled[valid_indices]
        
        test_mse = mean_squared_error(y_test_valid, y_pred_valid)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_valid, y_pred_valid)
        test_r2 = r2_score(y_test_valid, y_pred_valid)
        
        # CI coverage
        lower_bound = y_pred_valid - 1.96 * y_std_valid
        upper_bound = y_pred_valid + 1.96 * y_std_valid
        within_ci = ((y_test_valid >= lower_bound) & (y_test_valid <= upper_bound))
        ci_coverage = within_ci.mean() * 100
        
        test_metrics = {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'ci_coverage': ci_coverage,
            'n_valid': sum(valid_indices),
            'n_total': len(y_test_rescaled)
        }
    else:
        test_metrics = {
            'rmse': None,
            'mae': None,
            'r2': None,
            'ci_coverage': None,
            'n_valid': 0,
            'n_total': len(y_test_rescaled)
        }
        print("Warning: No valid test data points for metric calculation.")
    
    # Create test predictions DataFrame
    start_idx = len(data_for_model) - len(y_test) - seq_length
    end_idx = len(data_for_model) - seq_length
    test_indices = list(range(start_idx, end_idx))
    
    # Handle edge case where indices are out of bounds
    if start_idx < 0 or end_idx > len(data):
        test_predictions_df = pd.DataFrame()
    else:
        test_predictions_df = pd.DataFrame({
            "Month": data.iloc[test_indices]["month_num"].values,
            "Year": data.iloc[test_indices]["year"].values,
            "Date": [f"{y}-{m:02d}" for y, m in zip(data.iloc[test_indices]["year"], data.iloc[test_indices]["month_num"])],
            "Actual": y_test_rescaled,
            "Predicted": y_pred_mean_rescaled,
            "Uncertainty": y_pred_std_rescaled,
            "Lower_CI": lower_bound if 'lower_bound' in locals() else None,
            "Upper_CI": upper_bound if 'upper_bound' in locals() else None
        })
    
    # Find forecast start point
    target_date = pd.Timestamp(f"{forecast_start_year}-{forecast_start_month:02d}-01")
    closest_idx = (data['date'] - target_date).abs().idxmin()
    start_seq_idx = max(0, closest_idx - seq_length)
    forecast_start_sequence = data_scaled[start_seq_idx:start_seq_idx+seq_length]
    
    # Forecast future
    def forecast_future(model, last_sequence, n_steps, scaler, feature_indices, data):
        future_predictions = []
        prediction_std = []
        future_months = []
        future_years = []
        future_seasons = []
        future_gaps = []
        
        curr_sequence = last_sequence.copy()
        seq_length = curr_sequence.shape[0]
        n_features = curr_sequence.shape[1]
        
        last_row_unscaled = scaler.inverse_transform(curr_sequence[-1].reshape(1, -1))[0]
        curr_month = int(last_row_unscaled[feature_indices['month']])
        curr_year = int(last_row_unscaled[feature_indices['year']])
        
        for i in range(n_steps):
            # Predict next value
            pred_mean, pred_std = monte_carlo_predictions(
                model, curr_sequence.reshape(1, seq_length, n_features), n_simulations
            )
            
            # Update date
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1
            
            # Determine season from existing data
            matching_rows = data[data['month_num'] == curr_month]
            curr_season = matching_rows['Season_Index'].mean() if len(matching_rows) > 0 else 0.0
            
            # Always use 1 for months gap in forecast
            months_gap = 1
            
            # Store metadata
            future_months.append(curr_month)
            future_years.append(curr_year)
            future_seasons.append(curr_season)
            future_gaps.append(months_gap)
            
            # Create next row with predicted value
            temp_row = np.zeros((1, n_features))
            temp_row[0, feature_indices['year']] = curr_year
            temp_row[0, feature_indices['month']] = curr_month
            temp_row[0, feature_indices['season']] = curr_season
            temp_row[0, feature_indices['gap']] = 1  # Always use 1 for forecast gaps
            
            # Scale the row
            temp_row_scaled = scaler.transform(temp_row)[0]
            
            # Create full prediction row
            pred_full = np.zeros((1, n_features))
            for j in range(n_features):
                if j == feature_indices['precip']:
                    pred_full[0, j] = pred_mean[0, 0]  # Predicted precipitation
                else:
                    pred_full[0, j] = temp_row_scaled[j]
            
            # Update sequence for next prediction
            curr_sequence = np.append(curr_sequence[1:], pred_full, axis=0)
            
            # Store predictions
            future_predictions.append(pred_mean[0, 0])
            prediction_std.append(pred_std[0, 0])
        
        # Convert to arrays
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        prediction_std = np.array(prediction_std).reshape(-1, 1)
        
        # Rescale
        future_pred_rescaled, future_std_rescaled = rescale_predictions(
            future_predictions, prediction_std, None, feature_indices['precip'], scaler, n_features
        )
        
        return future_pred_rescaled, future_std_rescaled, future_months, future_years, future_seasons, future_gaps
    
    # Generate forecast
    future_pred, future_std, future_months, future_years, future_seasons, future_gaps = forecast_future(
        model, forecast_start_sequence, n_future, scaler, feature_indices, data
    )
    
    # Create forecast DataFrame
    future_dates = [f"{year}-{month:02d}" for year, month in zip(future_years, future_months)]
    future_date_objects = [pd.Timestamp(f"{year}-{month:02d}-01") for year, month in zip(future_years, future_months)]
    
    # Find actual values where available
    future_actuals = []
    for year, month in zip(future_years, future_months):
        matching_rows = data[(data["year"] == year) & (data["month_num"] == month)]
        if len(matching_rows) > 0:        
            actual_value = matching_rows["precipitation"].values[0]
            future_actuals.append(actual_value)
        else:
            future_actuals.append(None)
    
    # Create forecast DataFrame
    if len(future_pred) > 0:
        forecast_df = pd.DataFrame({
            'Month': future_months,
            'Year': future_years,
            'Season_Index': future_seasons,
            'Months_Gap': future_gaps,
            'Date': future_dates,
            'Date_Object': future_date_objects,
            'Forecasted Precipitation': future_pred,
            'Actual_if_available': future_actuals,
            'Forecast Uncertainty (Std Dev)': future_std,
            'Lower_Bound_95': future_pred - 1.96 * future_std,
            'Upper_Bound_95': future_pred + 1.96 * future_std
        })
    else:
        # Create empty DataFrame with the right columns if no forecasts
        forecast_df = pd.DataFrame(columns=[
            'Month', 'Year', 'Season_Index', 'Months_Gap', 'Date', 'Date_Object',
            'Forecasted Precipitation', 'Actual_if_available', 'Forecast Uncertainty (Std Dev)',
            'Lower_Bound_95', 'Upper_Bound_95'
        ])
    
    # Add error and percent error columns
    if not forecast_df.empty:
        forecast_df['Error'] = None
        forecast_df['Percent_Error'] = None
        
        for i in range(len(forecast_df)):
            if forecast_df['Actual_if_available'].iloc[i] is not None:
                forecast_df['Error'].iloc[i] = (forecast_df['Actual_if_available'].iloc[i] - 
                                             forecast_df['Forecasted Precipitation'].iloc[i])
                if forecast_df['Actual_if_available'].iloc[i] != 0:  # Avoid division by zero
                    forecast_df['Percent_Error'].iloc[i] = (forecast_df['Error'].iloc[i] / 
                                                         forecast_df['Actual_if_available'].iloc[i]) * 100
        
    # Calculate forecast metrics where actuals are available
    forecast_metrics = {}
    
    if not forecast_df.empty:
        forecast_with_actuals = forecast_df.dropna(subset=['Actual_if_available'])
        
        if len(forecast_with_actuals) > 0:
            actual_values = forecast_with_actuals['Actual_if_available'].values
            predicted_values = forecast_with_actuals['Forecasted Precipitation'].values
            
            forecast_mse = mean_squared_error(actual_values, predicted_values)
            forecast_rmse = np.sqrt(forecast_mse)
            forecast_mae = mean_absolute_error(actual_values, predicted_values)
            forecast_r2 = r2_score(actual_values, predicted_values)
            
            within_ci = ((forecast_with_actuals['Actual_if_available'] >= forecast_with_actuals['Lower_Bound_95']) &
                         (forecast_with_actuals['Actual_if_available'] <= forecast_with_actuals['Upper_Bound_95']))
            ci_coverage_forecast = within_ci.mean() * 100
            
            forecast_metrics = {
                'rmse': forecast_rmse,
                'mae': forecast_mae,
                'r2': forecast_r2,
                'ci_coverage': ci_coverage_forecast,
                'n_with_actuals': len(forecast_with_actuals)
            }
        else:
            forecast_metrics = {
                'rmse': None,
                'mae': None,
                'r2': None, 
                'ci_coverage': None,
                'n_with_actuals': 0
            }
    
    # Return results
    return {
        'model': model,  # Trained model for later use
        'scaler': scaler,  # Scaler for future predictions
        
        # Test evaluation results
        'test_metrics': {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'ci_coverage': ci_coverage
        },
        'test_predictions_df': test_predictions_df,
        
        # Forecast results
        'forecast_df': forecast_df,
        'forecast_metrics': forecast_metrics,
        
        # Training info
        'training_epochs': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1]
    }

def MilkProductionForecaster(data_numeric, features, unique_ward1="Shapefile_wardName", target_var="amountmilked", 
                           seq_length=13, test_size=0.1, epochs=100, batch_size=16,
                           n_future=16, external_precip_forecasts=None, n_simulations=100):
    """
    Complete milk production forecasting pipeline with Ward information
    
    Parameters:
    -----------
    data_numeric : DataFrame
        Input data containing all features
    features : list
        List of feature names to use for modeling
    unique_ward1 : str
        Name of the Shapefile_wardName column (default: "Shapefile_wardName")
    target_var : str, optional
        Name of the target variable column (default: "amountmilked")
    seq_length : int, optional
        Length of sequences for LSTM (default: 13)
    test_size : float, optional
        Proportion of data to use for testing (default: 0.1)
    epochs : int, optional
        Maximum number of training epochs (default: 100)
    batch_size : int, optional
        Batch size for training (default: 16)
    n_future : int, optional
        Number of future periods to forecast (default: 16)
    external_precip_forecasts : dict or DataFrame, optional
        External precipitation forecasts (default: None)
    n_simulations : int, optional
        Number of Monte Carlo simulations (default: 100)
        
    Returns:
    --------
    dict
        Dictionary containing all results including:
        - model: Trained Keras model
        - scaler: Fitted MinMaxScaler
        - training_history: Training history
        - evaluation_metrics: Dictionary of evaluation metrics
        - test_results: DataFrame with test predictions (includes Ward)
        - forecast_results: DataFrame with future forecasts (includes Ward)
        - feature_indices: Dictionary of feature indices
        - data_month_to_season: Dictionary mapping months to seasons
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    import json
    
    # Basic error handling - check if data is empty
    if data_numeric is None or data_numeric.empty:
        # Return empty results
        return {
            'model': None,
            'scaler': None,
            'training_history': None,
            'evaluation_metrics': {},
            'test_results': pd.DataFrame(),
            'forecast_results': pd.DataFrame(),
            'feature_indices': {},
            'data_month_to_season': {}
        }
    
    # Prepare data
    data = data_numeric.copy()
    data = data.sort_values(['year', 'month_num'])
    data['month_year'] = data['year'] * 12 + data['month_num']
    data['months_gap'] = data['month_year'].diff().fillna(1).astype(int)
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + 
                                 data['month_num'].astype(str).str.zfill(2) + '-01')
    
    # Store month to season mapping
    data_month_to_season = {}
    for month in range(1, 13):
        month_data = data[data['month_num'] == month]
        if len(month_data) > 0:
            data_month_to_season[month] = month_data['Season_Index'].mean()
    
    # Get feature indices
    feature_indices = {feature: idx for idx, feature in enumerate(features)}
    target_var_idx = feature_indices[target_var]
    
    # Scale data
    try:
        data_for_scaling = data[features].copy()
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_for_scaling)
    except Exception as e:
        print(f"Error during data scaling: {str(e)}")
        return {
            'model': None,
            'scaler': None,
            'training_history': None,
            'evaluation_metrics': {},
            'test_results': pd.DataFrame(),
            'forecast_results': pd.DataFrame(),
            'feature_indices': feature_indices,
            'data_month_to_season': data_month_to_season
        }
    
    # Prepare sequences
    def preprocess_data(data, target_idx, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len, :])
            y.append(data[i + seq_len, target_idx])
        return np.array(X), np.array(y)
    
    X, y = preprocess_data(data_scaled, target_var_idx, seq_length)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Build model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Monte Carlo predictions function
    def monte_carlo_predictions(model, X_input, n_simulations):
        @tf.function
        def f_model_mc(X_input, training=True):
            return model(X_input, training=training)
        
        preds = np.array([f_model_mc(X_input, training=True).numpy() 
                         for _ in range(n_simulations)])
        return preds.mean(axis=0), preds.std(axis=0)
    
    # Evaluate on test set
    y_pred_mean, y_pred_std = monte_carlo_predictions(model, X_test, n_simulations)
    
    # Rescale predictions
    def rescale_predictions(pred_mean, pred_std, y_true):
        dummy_pred = np.zeros((len(pred_mean), len(features)))
        dummy_pred[:, target_var_idx] = pred_mean.flatten()
        pred_mean_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]
        
        dummy_std = np.zeros((len(pred_std), len(features)))
        dummy_std[:, target_var_idx] = pred_std.flatten()
        pred_std_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]
        
        dummy_y = np.zeros((len(y_true), len(features)))
        dummy_y[:, target_var_idx] = y_true
        y_true_rescaled = scaler.inverse_transform(dummy_y)[:, target_var_idx]
        
        return pred_mean_rescaled, pred_std_rescaled, y_true_rescaled
    
    y_pred_mean_rescaled, y_pred_std_rescaled, y_test_rescaled = rescale_predictions(
        y_pred_mean, y_pred_std, y_test
    )
    
    # Calculate metrics
    mse = mean_squared_error(y_test_rescaled, y_pred_mean_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, y_pred_mean_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_mean_rescaled)
    
    # Calculate confidence interval coverage
    within_ci = (
        (y_test_rescaled >= (y_pred_mean_rescaled - 1.96 * y_pred_std_rescaled)) & 
        (y_test_rescaled <= (y_pred_mean_rescaled + 1.96 * y_pred_std_rescaled))
    )
    ci_coverage = within_ci.mean() * 100
    
    # Create evaluation metrics dictionary
    evaluation_metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ci_coverage': ci_coverage
    }
    
    # Convert evaluation metrics to string for inclusion in DataFrame
    evaluation_metrics_str = json.dumps(evaluation_metrics)
    
    # Prepare test results with Ward information
    test_indices = list(range(len(data) - len(y_test_rescaled) - seq_length, 
                            len(data) - seq_length))
    
    test_results = pd.DataFrame({
        'Ward': unique_ward1,
        'Month': data.iloc[test_indices]["month_num"].values,
        'Year': data.iloc[test_indices]["year"].values,
        'Season_Index': data.iloc[test_indices]["Season_Index"].values,
        'Precipitation': data.iloc[test_indices]["precipitation"].values,
        'Months Gap': data.iloc[test_indices]["months_gap"].values,
        'Date': [f"{year}-{month:02d}" for year, month in 
                zip(data.iloc[test_indices]["year"].values, 
                   data.iloc[test_indices]["month_num"].values)],
        'Date_Object': data.iloc[test_indices]["date"].values,
        'Actual': y_test_rescaled,
        'Forecasted Amount Milked': y_pred_mean_rescaled,
        'Forecast Uncertainty (Std Dev)': y_pred_std_rescaled,
        'Lower Bound (95%)': y_pred_mean_rescaled - 1.96 * y_pred_std_rescaled,
        'Upper Bound (95%)': y_pred_mean_rescaled + 1.96 * y_pred_std_rescaled,
        'Error': y_test_rescaled - y_pred_mean_rescaled,
        'Percent Error': ((y_test_rescaled - y_pred_mean_rescaled) / y_test_rescaled) * 100
    })

    # Future forecasting
    def forecast_future(last_sequence, n_future):
        curr_sequence = last_sequence.copy()
        future_pred = []
        future_std = []
        future_months = []
        future_years = []
        future_seasons = []
        future_precip = []
        future_forecast_precip = []
        future_gaps = []
        
        # Prepare precipitation forecasts
        precip_forecast = {}
        if external_precip_forecasts is not None:
            if isinstance(external_precip_forecasts, pd.DataFrame):
                for _, row in external_precip_forecasts.iterrows():
                    precip_forecast[(row['year'], row['month_num'])] = row['Forecasted Precipitation']
            elif isinstance(external_precip_forecasts, dict):
                precip_forecast = external_precip_forecasts.copy()
        
        # Initialize current values
        last_row_unscaled = scaler.inverse_transform(curr_sequence[-1].reshape(1, -1))[0]
        curr_month = int(last_row_unscaled[feature_indices['month_num']])
        curr_year = int(last_row_unscaled[feature_indices['year']])
        
        for _ in range(n_future):
            # Make prediction
            pred_mean, pred_std = monte_carlo_predictions(
                model, curr_sequence.reshape(1, seq_length, len(features)), n_simulations
            )
            
            # Update month/year
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1
            
            # Get season
            curr_season = data_month_to_season.get(curr_month, 0.0)
            
            # Get precipitation
            future_key = (curr_year, curr_month)
            if future_key in precip_forecast:
                curr_forecast_precip = precip_forecast[future_key]
            else:
                month_data = data[data['month_num'] == curr_month]
                if len(month_data) > 0:
                    curr_forecast_precip = month_data['Forecasted Precipitation'].mean()
                else:
                    curr_forecast_precip = 0.0
            
            # Get historical precipitation
            month_data = data[data['month_num'] == curr_month]
            if len(month_data) > 0:
                curr_precip = month_data['precipitation'].mean()
            else:
                curr_precip = 0.0
            
            # Store values
            future_months.append(curr_month)
            future_years.append(curr_year)
            future_seasons.append(curr_season)
            future_precip.append(curr_precip)
            future_forecast_precip.append(curr_forecast_precip)
            future_gaps.append(1)
            
            # Create new row for next prediction
            temp_row = np.zeros((1, len(features)))
            temp_row[0, feature_indices['year']] = curr_year
            temp_row[0, feature_indices['month_num']] = curr_month
            temp_row[0, feature_indices['Season_Index']] = curr_season
            temp_row[0, feature_indices['precipitation']] = curr_precip
            temp_row[0, feature_indices['Forecasted Precipitation']] = curr_forecast_precip
            temp_row[0, feature_indices['months_gap']] = 1
            
            # Scale and update sequence
            temp_row_scaled = scaler.transform(temp_row)[0]
            pred_full = np.zeros((1, len(features)))
            for j in range(len(features)):
                if j == target_var_idx:
                    pred_full[0, j] = pred_mean[0, 0]
                else:
                    pred_full[0, j] = temp_row_scaled[j]
            
            curr_sequence = np.append(curr_sequence[1:], pred_full, axis=0)
            
            # Store predictions
            future_pred.append(pred_mean[0, 0])
            future_std.append(pred_std[0, 0])
        
        # Rescale predictions
        future_pred = np.array(future_pred).reshape(-1, 1)
        future_std = np.array(future_std).reshape(-1, 1)
        
        dummy_pred = np.zeros((len(future_pred), len(features)))
        dummy_pred[:, target_var_idx] = future_pred.flatten()
        future_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]
        
        dummy_std = np.zeros((len(future_std), len(features)))
        dummy_std[:, target_var_idx] = future_std.flatten()
        future_std_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]
        
        return (future_pred_rescaled, future_std_rescaled, future_months, future_years,
                future_seasons, future_precip, future_forecast_precip, future_gaps)
    
    # Generate future forecasts
    future_pred, future_std, future_months, future_years, future_seasons, \
    future_precip, future_forecast_precip, future_gaps = forecast_future(X_test[-1], n_future)
    
    # Prepare forecast results with Ward information
    future_dates = [f"{year}-{month:02d}" for year, month in zip(future_years, future_months)]
    future_date_objects = [pd.Timestamp(f"{year}-{month:02d}-01") for year, month in zip(future_years, future_months)]
    
    # Get actuals if available
    future_actuals = []
    for year, month in zip(future_years, future_months):
        matching_rows = data[(data["year"] == year) & (data["month_num"] == month)]
        if len(matching_rows) > 0:
            future_actuals.append(matching_rows[target_var].values[0])
        else:
            future_actuals.append(None)
    
    forecast_results = pd.DataFrame({
        'Ward': unique_ward1,
        'Month': future_months,
        'Year': future_years,
        'Season_Index': future_seasons,
        'Precipitation': future_precip,
        'Months Gap': future_gaps,
        'Date': future_dates,
        'Date_Object': future_date_objects,
        'Forecasted Amount Milked': future_pred,
        'Actual (if available)': future_actuals,
        'Forecast Uncertainty (Std Dev)': future_std,
        'Lower Bound (95%)': future_pred - 1.96 * future_std,
        'Upper Bound (95%)': future_pred + 1.96 * future_std,
        'Error': [None if actual is None else actual - pred for actual, pred in 
                 zip(future_actuals, future_pred)],
        'Percent Error': [None if actual is None or actual == 0 else ((actual - pred) / actual) * 100 
                         for actual, pred in zip(future_actuals, future_pred)]
    })
    
    # Find the last actual value in data_numeric
    try:
        # Get the last row with a non-null value for the target variable
        last_data = data_numeric.sort_values(['year', 'month_num'])
        last_actual_row = last_data[last_data[target_var].notna()].iloc[-1] if not last_data[last_data[target_var].notna()].empty else None
        
        # Add the 4 additional fields to forecast_results
        forecast_results['Last_Actual_Value'] = None
        forecast_results['Month1_Forecast'] = None
        forecast_results['Month2_Forecast'] = None
        forecast_results['Month3_Forecast'] = None
        forecast_results['Evaluation_Metrics'] = None  # Add the metrics column
        
        if last_actual_row is not None:
            # Get the date information for the last actual
            last_year = last_actual_row['year']
            last_month = last_actual_row['month_num']
            last_actual_value = last_actual_row[target_var]
            
            # Find the row in forecast_results that corresponds to the last actual date
            last_actual_mask = (forecast_results['Year'] == last_year) & (forecast_results['Month'] == last_month)
            
            if last_actual_mask.any():
                # Get the index of the row with the last actual value
                last_idx = forecast_results[last_actual_mask].index[0]
                
                # Set the last actual value
                forecast_results.loc[last_idx, 'Last_Actual_Value'] = last_actual_value
                
                # Set the evaluation metrics for just the last actual row
                forecast_results.loc[last_idx, 'Evaluation_Metrics'] = evaluation_metrics_str
                
                # Get the next 3 month forecasts (if they exist)
                if last_idx + 1 < len(forecast_results):
                    forecast_results.loc[last_idx, 'Month1_Forecast'] = forecast_results.loc[last_idx + 1, 'Forecasted Amount Milked']
                    
                if last_idx + 2 < len(forecast_results):
                    forecast_results.loc[last_idx, 'Month2_Forecast'] = forecast_results.loc[last_idx + 2, 'Forecasted Amount Milked']
                    
                if last_idx + 3 < len(forecast_results):
                    forecast_results.loc[last_idx, 'Month3_Forecast'] = forecast_results.loc[last_idx + 3, 'Forecasted Amount Milked']
    except Exception as e:
        print(f"Error adding last actual value fields: {str(e)}")
    
    # Return all results
    return {
        'model': model,
        'scaler': scaler,
        'training_history': history.history,
        'evaluation_metrics': evaluation_metrics,
        'test_results': test_results,
        'forecast_results': forecast_results,
        'feature_indices': feature_indices,
        'data_month_to_season': data_month_to_season
    }
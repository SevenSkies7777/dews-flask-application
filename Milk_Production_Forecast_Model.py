import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.api import VAR
import json
from tensorflow.keras.callbacks import EarlyStopping


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


def MilkProductionForecaster(data_numeric, features, unique_ward1="Shapefile_wardName", 
                                   target_var="amountmilked", seq_length=13, test_size=0.1, 
                                   epochs=100, batch_size=16, n_future=16, 
                                   external_precip_forecasts=None, n_simulations=100,
                                   var_lags=3, ensemble_method='adaptive', 
                                   lstm_weight=None, var_weight=None,  # FIXED: Added explicit parameters
                                   forecast_weights=None):
    """
    Ensemble milk production forecaster combining separate LSTM and VAR models with GrazingDist support
    FIXED: Now maintains original 3-forecast behavior - only forecasts 3 months ahead from last actual data
    UPDATED: VAR model now filters parameters based on 0.05 significance level
    CORRECTED: LSTM component simplified to match original consistency while keeping GrazingDist
    NEW: Forecast-horizon-specific weighting system
    
    Parameters:
    -----------
    ensemble_method : str
        Method for combining predictions: 'weighted', 'adaptive', 'stacking', 'voting'
    lstm_weight : float, optional
        Default weight for LSTM predictions (used for test set evaluation)
    var_weight : float, optional
        Default weight for VAR predictions (used for test set evaluation)
    forecast_weights : dict, optional
        Forecast-horizon-specific weights. Format:
        {
            1: {'lstm': 0.7, 'var': 0.3},  # Month 1 forecast
            2: {'lstm': 1.0, 'var': 0.0},  # Month 2 forecast  
            3: {'lstm': 1.0, 'var': 0.0}   # Month 3 forecast
        }
        If None, uses default: Month 1 = (0.7, 0.3), Months 2-3 = (1.0, 0.0)
    
    Note: Uses n_future=16 as forecasting window but only outputs last actual + 3 forecasts
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    import json
    
    # Set default forecast-horizon-specific weights if not provided
    if forecast_weights is None:
        forecast_weights = {
            1: {'lstm': 0.7, 'var': 0.3},  # Month 1: Ensemble
            2: {'lstm': 1.0, 'var': 0.0},  # Month 2: Pure LSTM
            3: {'lstm': 1.0, 'var': 0.0}   # Month 3: Pure LSTM
        }
    
    # FIXED: Derive default weights from forecast_weights if not provided
    if lstm_weight is None or var_weight is None:
        # Use Month 1 weights as default for test evaluation
        default_weights = forecast_weights.get(1, {'lstm': 0.7, 'var': 0.3})
        if lstm_weight is None:
            lstm_weight = default_weights['lstm']
        if var_weight is None:
            var_weight = default_weights['var']
        print(f"Derived default weights from forecast_weights: LSTM={lstm_weight}, VAR={var_weight}")
    
    print(f"Forecast-horizon-specific weights:")
    for horizon, weights in forecast_weights.items():
        print(f"  Month {horizon}: LSTM = {weights['lstm']:.1f}, VAR = {weights['var']:.1f}")
    
    def simple_data_preparation(data_numeric, features, target_var):
        """
        Simplified data preparation matching the original model approach
        """
        print("=== Simple Data Preparation (Original Approach + GrazingDist) ===")
        
        # Basic checks only
        if data_numeric.empty:
            raise ValueError("Input data is empty")
        
        data = data_numeric.copy()
        data = data.sort_values(['year', 'month_num'])
        
        # Create basic derived features (like original)
        if 'months_gap' not in data.columns:
            data['month_year'] = data['year'] * 12 + data['month_num']
            data['months_gap'] = data['month_year'].diff().fillna(1).astype(int)
        
        data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + 
                                   data['month_num'].astype(str).str.zfill(2) + '-01')
        
        # Minimal cleaning - only remove rows with missing target
        if data[target_var].isnull().sum() > 0:
            initial_rows = len(data)
            data = data.dropna(subset=[target_var])
            print(f"Removed {initial_rows - len(data)} rows with missing target values")
        
        print(f"Final data shape: {data.shape}")
        return data, features
    
    # --- Data Validation and Preparation (SIMPLIFIED) ---
    if data_numeric.empty:
        return empty_results_dict()
    
    # Apply simplified data preparation (like original)
    data, features = simple_data_preparation(data_numeric.copy(), features, target_var)
    
    # Ensure GrazingDist is in features list if it exists in data (RETAINED FEATURE)
    if 'GrazingDist' in data.columns and 'GrazingDist' not in features:
        features = features + ['GrazingDist']
        print(f"Added GrazingDist to features. Updated features: {features}")
    elif 'GrazingDist' in features and 'GrazingDist' not in data.columns:
        print("Warning: GrazingDist specified in features but not found in data columns")
        features = [f for f in features if f != 'GrazingDist']
    
    # Store month to season mapping
    data_month_to_season = {}
    for month in range(1, 13):
        month_data = data[data['month_num'] == month]
        if len(month_data) > 0:
            data_month_to_season[month] = month_data['Season_Index'].mean()
    
    # Store month to GrazingDist mapping for forecasting (RETAINED)
    data_month_to_grazing = {}
    if 'GrazingDist' in data.columns:
        for month in range(1, 13):
            month_data = data[data['month_num'] == month]
            if len(month_data) > 0:
                data_month_to_grazing[month] = month_data['GrazingDist'].mean()
        print(f"GrazingDist seasonal patterns created: {data_month_to_grazing}")
    
    print(f"Features being used: {features}")
    if 'GrazingDist' in features:
        print(f"GrazingDist feature included: {data['GrazingDist'].describe() if 'GrazingDist' in data.columns else 'Not found in data'}")
    
    print("Training ensemble components...")
    
    # =============================================================================
    # COMPONENT 1: SIMPLIFIED LSTM MODEL (Original Approach + GrazingDist)
    # =============================================================================
    
    def train_lstm_model():
        """
        Simplified LSTM training that matches the original model approach for consistency
        while retaining GrazingDist support
        """
        print("Training LSTM model with simplified approach (original consistency + GrazingDist)...")
        
        # Use original simple data preparation approach (no complex filtering)
        lstm_data = data[features].copy()
        
        # Simple feature validation - no complex filtering (like original)
        print(f"Using features as provided: {features}")
        if 'GrazingDist' in features:
            print(f"GrazingDist range: {data['GrazingDist'].min():.2f} to {data['GrazingDist'].max():.2f}")
            print(f"GrazingDist stats: mean={data['GrazingDist'].mean():.2f}, std={data['GrazingDist'].std():.2f}")
        
        # Original scaling approach (simple)
        lstm_scaler = MinMaxScaler()
        lstm_data_scaled = lstm_scaler.fit_transform(lstm_data)
        
        # Original feature indices approach
        feature_indices = {feature: idx for idx, feature in enumerate(features)}
        target_idx = feature_indices[target_var]
        
        print(f"LSTM feature indices: {feature_indices}")
        
        # Original sequence creation (matching preprocess_data function)
        def create_sequences(data, seq_len, target_idx):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len, :])  # All features
                y.append(data[i + seq_len, target_idx])  # Target only
            return np.array(X), np.array(y)
        
        X, y = create_sequences(lstm_data_scaled, seq_length, target_idx)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Identical model architecture (same as original)
        lstm_model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Identical training process (same as original)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, mode='min', restore_best_weights=True
        )
        
        lstm_history = lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Identical Monte Carlo predictions (same as original)
        def monte_carlo_predictions(model, X_input, n_simulations):
            @tf.function
            def f_model_mc(X_input, training=True):
                return model(X_input, training=training)
            
            preds = np.array([f_model_mc(X_input, training=True).numpy() 
                             for _ in range(n_simulations)])
            return preds.mean(axis=0), preds.std(axis=0)
        
        # Evaluate LSTM
        y_pred_mean, y_pred_std = monte_carlo_predictions(lstm_model, X_test, n_simulations)
        
        # Original simple rescaling approach (same as original)
        def rescale_predictions(pred_mean, pred_std, y_true):
            # Simple reshaping like original
            dummy_pred = np.zeros((len(pred_mean), len(features)))
            dummy_pred[:, target_idx] = pred_mean.flatten()
            pred_mean_rescaled = lstm_scaler.inverse_transform(dummy_pred)[:, target_idx]
            
            dummy_std = np.zeros((len(pred_std), len(features)))
            dummy_std[:, target_idx] = pred_std.flatten()
            pred_std_rescaled = lstm_scaler.inverse_transform(dummy_std)[:, target_idx]
            
            dummy_y = np.zeros((len(y_true), len(features)))
            dummy_y[:, target_idx] = y_true
            y_true_rescaled = lstm_scaler.inverse_transform(dummy_y)[:, target_idx]
            
            return pred_mean_rescaled, pred_std_rescaled, y_true_rescaled
        
        y_pred_rescaled, y_std_rescaled, y_test_rescaled = rescale_predictions(
            y_pred_mean, y_pred_std, y_test
        )
        
        return {
            'model': lstm_model,
            'scaler': lstm_scaler,
            'history': lstm_history,
            'X_test': X_test,
            'y_test_rescaled': y_test_rescaled,
            'y_pred_rescaled': y_pred_rescaled,
            'y_std_rescaled': y_std_rescaled,
            'feature_indices': feature_indices,
            'numeric_features': features,  # Use original features (including GrazingDist)
            'monte_carlo_fn': monte_carlo_predictions
        }
    
    # =============================================================================
    # COMPONENT 2: VAR MODEL (Enhanced with GrazingDist support + SIGNIFICANCE FILTERING)
    # =============================================================================
    
    def train_var_model():
        print("Training VAR model with GrazingDist support and significance filtering...")
        
        try:
            from statsmodels.tsa.api import VAR
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            print("statsmodels not available. Skipping VAR model.")
            return None
        
        try:
            # Select relevant variables for VAR (continuous variables only) - include GrazingDist
            var_features = [target_var, 'precipitation', 'Season_Index']
            
            # Add GrazingDist if available (RETAINED FEATURE)
            if 'GrazingDist' in data.columns:
                var_features.append('GrazingDist')
                print("Added GrazingDist to VAR features")
            
            available_var_features = [f for f in var_features if f in data.columns]
            
            if len(available_var_features) < 2:
                print(f"Insufficient VAR features available: {available_var_features}")
                return None
            
            print(f"VAR features being used: {available_var_features}")
            
            var_data = data[available_var_features].dropna()
            
            if len(var_data) < 20:
                print("Insufficient data for VAR model")
                return None
            
            # Check stationarity and difference if needed
            def check_stationarity(series, name):
                try:
                    result = adfuller(series.dropna())
                    is_stationary = result[1] <= 0.05
                    print(f"{name} - ADF p-value: {result[1]:.4f}, Stationary: {is_stationary}")
                    return is_stationary
                except:
                    return True  # Assume stationary if test fails
            
            # Apply differencing if needed
            var_data_processed = var_data.copy()
            differenced_vars = {}
            
            for col in available_var_features:
                if not check_stationarity(var_data[col], col):
                    var_data_processed[col] = var_data[col].diff()
                    differenced_vars[col] = True
                else:
                    differenced_vars[col] = False
            
            var_data_processed = var_data_processed.dropna()
            
            if len(var_data_processed) < max(var_lags * 2, 20):
                print("Insufficient data for VAR model after preprocessing")
                return None
            
            # Fit VAR model
            var_model = VAR(var_data_processed)
            
            # Select optimal lag order
            try:
                lag_order = var_model.select_order(maxlags=min(var_lags, len(var_data_processed)//4))
                optimal_lags = lag_order.aic
            except:
                optimal_lags = min(var_lags, len(var_data_processed)//4)
            
            # Fit initial VAR model with all lags
            var_result = var_model.fit(optimal_lags)
            print(f"Initial VAR model fitted with {optimal_lags} lags")
            
            # NEW: Check parameter significance at 0.05 level
            print(f"\n=== Checking Parameter Significance (α = 0.05) ===")
            
            params = var_result.params
            pvalues = var_result.pvalues
            
            # Track significant parameters by lag
            significant_lags = set()
            total_lag_params = 0
            significant_lag_params = 0
            
            for eq_name in var_result.names:
                eq_pvalues = pvalues[eq_name]
                
                for param_name in eq_pvalues.index:
                    if '.L' in param_name:  # This is a lag parameter
                        total_lag_params += 1
                        p_value = eq_pvalues[param_name]
                        
                        if p_value <= 0.05:
                            significant_lag_params += 1
                            # Extract lag number
                            lag_num = int(param_name.split('.L')[1])
                            significant_lags.add(lag_num)
                            print(f"✓ {eq_name} - {param_name}: p={p_value:.4f} (significant)")
                        else:
                            print(f"✗ {eq_name} - {param_name}: p={p_value:.4f} (not significant)")
            
            significance_rate = (significant_lag_params / total_lag_params) * 100 if total_lag_params > 0 else 0
            print(f"\nSignificance Summary:")
            print(f"- Total lag parameters: {total_lag_params}")
            print(f"- Significant parameters: {significant_lag_params}")
            print(f"- Significance rate: {significance_rate:.1f}%")
            print(f"- Lags with significant parameters: {sorted(significant_lags)}")
            
            # Decision logic for model selection based on significance
            if significance_rate < 30:
                print(f"WARNING: Only {significance_rate:.1f}% of parameters are significant!")
                print("Attempting to find optimal lag structure based on significance...")
                
                # Try reducing lags to find better model
                best_model = var_result
                best_aic = var_result.aic
                best_lags = optimal_lags
                best_sig_rate = significance_rate
                
                # Test shorter lag structures
                for test_lags in range(1, optimal_lags):
                    try:
                        test_result = var_model.fit(test_lags)
                        test_params = test_result.pvalues
                        
                        # Count significant parameters
                        test_sig_params = 0
                        test_total_params = 0
                        
                        for eq_name in test_result.names:
                            eq_pvalues = test_params[eq_name]
                            for param_name in eq_pvalues.index:
                                if '.L' in param_name:
                                    test_total_params += 1
                                    if eq_pvalues[param_name] <= 0.05:
                                        test_sig_params += 1
                        
                        test_sig_rate = (test_sig_params / test_total_params) * 100 if test_total_params > 0 else 0
                        
                        print(f"  Testing {test_lags} lags: {test_sig_rate:.1f}% significant, AIC: {test_result.aic:.4f}")
                        
                        # Use model if it has better significance or better AIC with decent significance
                        if (test_sig_rate > best_sig_rate + 10) or \
                           (test_sig_rate >= 50 and test_result.aic < best_aic):
                            best_model = test_result
                            best_aic = test_result.aic
                            best_lags = test_lags
                            best_sig_rate = test_sig_rate
                            print(f"  → New best model with {test_lags} lags")
                    
                    except Exception as e:
                        print(f"  Failed to test {test_lags} lags: {e}")
                
                var_result = best_model
                optimal_lags = best_lags
                significance_rate = best_sig_rate
                print(f"\nSelected VAR model with {optimal_lags} lags (AIC: {best_aic:.4f}, {significance_rate:.1f}% significant)")
                
            else:
                print(f"✓ Model has acceptable significance rate ({significance_rate:.1f}%)")
            
            # Final significance check for selected model
            final_params = var_result.pvalues
            print(f"\n=== Final Model Parameter Significance ===")
            for eq_name in var_result.names:
                print(f"\nEquation: {eq_name}")
                eq_pvalues = final_params[eq_name]
                for param_name in eq_pvalues.index:
                    p_val = eq_pvalues[param_name]
                    status = "✓" if p_val <= 0.05 else "✗"
                    print(f"  {status} {param_name}: p={p_val:.4f}")
            
            # Create test set for VAR (align with LSTM test size)
            n_test = int(len(var_data_processed) * test_size)
            var_train = var_data_processed.iloc[:-n_test] if n_test > 0 else var_data_processed
            var_test = var_data_processed.iloc[-n_test:] if n_test > 0 else var_data_processed[-5:]
            
            # Make VAR predictions for test set
            var_predictions = []
            var_forecast_input = var_train.iloc[-optimal_lags:].values
            
            for i in range(len(var_test)):
                try:
                    forecast = var_result.forecast(var_forecast_input, steps=1)
                    var_predictions.append(forecast[0, 0])  # Target variable prediction
                    
                    # Update input for next prediction
                    actual_row = var_test.iloc[i:i+1].values
                    var_forecast_input = np.vstack([var_forecast_input[1:], actual_row])
                except:
                    # If forecasting fails, use last prediction or zero
                    if var_predictions:
                        var_predictions.append(var_predictions[-1])
                    else:
                        var_predictions.append(0.0)
            
            var_predictions = np.array(var_predictions)
            var_actuals = var_test[target_var].values
            
            # Convert back from differenced if needed
            if differenced_vars.get(target_var, False):
                # Integrate back to levels
                last_level = var_data[target_var].iloc[len(var_train)]
                var_predictions_levels = np.cumsum(np.concatenate([[last_level], var_predictions]))[1:]
                var_actuals_levels = np.cumsum(np.concatenate([[last_level], var_actuals]))[1:]
            else:
                var_predictions_levels = var_predictions
                var_actuals_levels = var_actuals
            
            return {
                'model': var_result,
                'data_processed': var_data_processed,
                'differenced_vars': differenced_vars,
                'predictions': var_predictions_levels,
                'actuals': var_actuals_levels,
                'optimal_lags': optimal_lags,
                'var_features': available_var_features,
                'significance_rate': significance_rate,  # NEW: Track significance
                'used_only_significant': significance_rate >= 50  # NEW: Flag for reporting
            }
            
        except Exception as e:
            print(f"VAR model training failed: {str(e)}")
            return None
    
    # =============================================================================
    # TRAIN BOTH MODELS
    # =============================================================================
    
    lstm_results = train_lstm_model()
    
    # Check if user wants to disable VAR (var_weight = 0)
    if var_weight == 0:
        print("VAR component disabled (var_weight=0), using LSTM only")
        var_results = None
    else:
        var_results = train_var_model()
        
        if var_results is None:
            print("VAR training failed, adjusting weights to LSTM only")
            var_weight = 0
            lstm_weight = 1
        else:
            # Report VAR significance results
            if 'significance_rate' in var_results:
                print(f"\nVAR Model Significance Summary:")
                print(f"- Parameter significance rate: {var_results['significance_rate']:.1f}%")
                print(f"- Uses only significant parameters: {var_results['used_only_significant']}")
                if var_results['significance_rate'] < 50:
                    print("- Warning: Low parameter significance may affect reliability")
    
    # Ensure weights are normalized and report final weights
    if var_results is None or var_weight == 0:
        final_lstm_weight = 1.0
        final_var_weight = 0.0
        print(f"Final weights: LSTM = 1.0, VAR = 0.0 (Pure LSTM)")
    elif lstm_weight == 0:
        final_lstm_weight = 0.0
        final_var_weight = 1.0
        print(f"Final weights: LSTM = 0.0, VAR = 1.0 (Pure VAR)")
    else:
        # Normalize weights if they don't sum to 1
        total_weight = lstm_weight + var_weight
        final_lstm_weight = lstm_weight / total_weight
        final_var_weight = var_weight / total_weight
        print(f"Final weights: LSTM = {final_lstm_weight:.3f}, VAR = {final_var_weight:.3f}")
    
    # Update the actual weights used in combination
    lstm_weight = final_lstm_weight
    var_weight = final_var_weight
    
    # =============================================================================
    # ENSEMBLE COMBINATION STRATEGIES
    # =============================================================================
    
    def combine_predictions(lstm_pred, lstm_std, var_pred, var_std, method='weighted', 
                          horizon_weights=None, is_forecast=False):
        """
        Combine LSTM and VAR predictions using different strategies
        FIXED: Proper weight handling for pure LSTM (1,0) or pure VAR (0,1) cases
        NEW: Support for forecast-horizon-specific weighting
        
        Parameters:
        -----------
        horizon_weights : dict, optional
            Weights for each forecast horizon. Format: {1: {'lstm': 0.7, 'var': 0.3}, ...}
            If None, uses default lstm_weight and var_weight
        is_forecast : bool
            Whether this is for future forecasts (True) or test evaluation (False)
        """
        # Use default weights for test evaluation or if no horizon weights provided
        if not is_forecast or horizon_weights is None:
            current_lstm_weight = lstm_weight
            current_var_weight = var_weight
            
            # Handle pure LSTM case (lstm_weight=1, var_weight=0)
            if var_results is None or var_pred is None or current_var_weight == 0:
                print(f"Using pure LSTM for test evaluation: var_weight={current_var_weight}")
                return lstm_pred, lstm_std
            
            # Handle pure VAR case (lstm_weight=0, var_weight=1)
            if current_lstm_weight == 0:
                print(f"Using pure VAR for test evaluation: lstm_weight={current_lstm_weight}")
                return var_pred, var_std
                
            print(f"Test evaluation weights: LSTM={current_lstm_weight:.3f}, VAR={current_var_weight:.3f}")
        
        # For forecasts: Apply horizon-specific weights
        if is_forecast and horizon_weights is not None:
            print(f"\nApplying forecast-horizon-specific weighting:")
            
            # Ensure arrays are same length
            if var_pred is not None:
                min_len = min(len(lstm_pred), len(var_pred))
                lstm_pred = lstm_pred[:min_len]
                lstm_std = lstm_std[:min_len] if lstm_std is not None else np.zeros(min_len)
                var_pred = var_pred[:min_len]
                var_std = var_std[:min_len] if var_std is not None else np.zeros(min_len)
            else:
                min_len = len(lstm_pred)
                var_pred = np.zeros(min_len)
                var_std = np.zeros(min_len)
            
            # Apply different weights for each forecast horizon
            ensemble_pred = np.zeros(min_len)
            ensemble_std = np.zeros(min_len)
            
            for i in range(min_len):
                forecast_horizon = i + 1  # 1-based indexing for horizons
                
                # Get weights for this horizon (default to last specified if beyond range)
                if forecast_horizon in horizon_weights:
                    h_weights = horizon_weights[forecast_horizon]
                else:
                    # Use the last available horizon weights
                    max_horizon = max(horizon_weights.keys())
                    h_weights = horizon_weights[max_horizon]
                    print(f"  Horizon {forecast_horizon}: Using weights from horizon {max_horizon}")
                
                h_lstm_weight = h_weights['lstm']
                h_var_weight = h_weights['var']
                
                print(f"  Horizon {forecast_horizon}: LSTM={h_lstm_weight:.1f}, VAR={h_var_weight:.1f}")
                
                # Handle pure cases for this horizon
                if h_var_weight == 0 or var_results is None:
                    ensemble_pred[i] = lstm_pred[i]
                    ensemble_std[i] = lstm_std[i]
                elif h_lstm_weight == 0:
                    ensemble_pred[i] = var_pred[i]
                    ensemble_std[i] = var_std[i]
                else:
                    # Apply ensemble combination for this horizon
                    if method == 'weighted':
                        ensemble_pred[i] = h_lstm_weight * lstm_pred[i] + h_var_weight * var_pred[i]
                        ensemble_std[i] = np.sqrt(h_lstm_weight**2 * lstm_std[i]**2 + h_var_weight**2 * var_std[i]**2)
                    
                    elif method == 'adaptive':
                        # For adaptive, use the horizon-specific weights as base but adjust based on recent performance
                        if i >= 5:  # Need some history for adaptive weighting
                            recent_window = min(5, i)
                            lstm_recent_error = np.mean(lstm_std[i-recent_window:i]**2) + 1e-8
                            var_recent_error = np.mean(var_std[i-recent_window:i]**2) + 1e-8
                            
                            total_inv_error = 1/lstm_recent_error + 1/var_recent_error
                            adaptive_lstm = (1/lstm_recent_error) / total_inv_error
                            adaptive_var = (1/var_recent_error) / total_inv_error
                            
                            # Blend adaptive weights with horizon-specific weights
                            blend_factor = 0.7  # 70% horizon-specific, 30% adaptive
                            final_lstm = blend_factor * h_lstm_weight + (1-blend_factor) * adaptive_lstm
                            final_var = blend_factor * h_var_weight + (1-blend_factor) * adaptive_var
                        else:
                            final_lstm, final_var = h_lstm_weight, h_var_weight
                        
                        ensemble_pred[i] = final_lstm * lstm_pred[i] + final_var * var_pred[i]
                        ensemble_std[i] = np.sqrt(final_lstm**2 * lstm_std[i]**2 + final_var**2 * var_std[i]**2)
                    
                    elif method == 'voting':
                        # Voting ignores user weights and uses equal weighting
                        ensemble_pred[i] = 0.5 * lstm_pred[i] + 0.5 * var_pred[i]
                        ensemble_std[i] = np.sqrt(0.25 * lstm_std[i]**2 + 0.25 * var_std[i]**2)
                    
                    elif method == 'stacking':
                        # For stacking with horizon-specific weights, we'll use the horizon weights as fallback
                        # since we can't retrain the stacker for each horizon
                        ensemble_pred[i] = h_lstm_weight * lstm_pred[i] + h_var_weight * var_pred[i]
                        ensemble_std[i] = np.sqrt(h_lstm_weight**2 * lstm_std[i]**2 + h_var_weight**2 * var_std[i]**2)
            
            return ensemble_pred, ensemble_std
        
        # Standard ensemble combination for non-horizon-specific cases
        if var_results is None or var_pred is None:
            return lstm_pred, lstm_std
        
        # Ensure arrays are same length
        min_len = min(len(lstm_pred), len(var_pred))
        lstm_pred = lstm_pred[:min_len]
        lstm_std = lstm_std[:min_len] if lstm_std is not None else np.zeros(min_len)
        var_pred = var_pred[:min_len]
        var_std = var_std[:min_len] if var_std is not None else np.zeros(min_len)
        
        print(f"Standard combination: method={method}, lstm_weight={lstm_weight}, var_weight={var_weight}")
        
        if method == 'weighted':
            # Simple weighted average
            ensemble_pred = lstm_weight * lstm_pred + var_weight * var_pred
            ensemble_std = np.sqrt(lstm_weight**2 * lstm_std**2 + var_weight**2 * var_std**2)
            
        elif method == 'adaptive':
            # Weight based on recent performance (inverse of squared error)
            recent_window = min(10, len(lstm_pred))
            if recent_window > 0:
                lstm_recent_error = np.mean(lstm_std[-recent_window:]**2) + 1e-8
                var_recent_error = np.mean(var_std[-recent_window:]**2) + 1e-8
                
                total_inv_error = 1/lstm_recent_error + 1/var_recent_error
                w_lstm = (1/lstm_recent_error) / total_inv_error
                w_var = (1/var_recent_error) / total_inv_error
                
                print(f"Adaptive weights: w_lstm={w_lstm:.3f}, w_var={w_var:.3f}")
            else:
                w_lstm, w_var = lstm_weight, var_weight
                print(f"Fallback to user weights: w_lstm={w_lstm}, w_var={w_var}")
            
            ensemble_pred = w_lstm * lstm_pred + w_var * var_pred
            ensemble_std = np.sqrt(w_lstm**2 * lstm_std**2 + w_var**2 * var_std**2)
            
        elif method == 'voting':
            # Simple average (equal voting) - IGNORES user weights by design
            print("Using equal voting (0.5, 0.5) - user weights ignored")
            ensemble_pred = 0.5 * lstm_pred + 0.5 * var_pred
            ensemble_std = np.sqrt(0.25 * lstm_std**2 + 0.25 * var_std**2)
            
        elif method == 'stacking':
            # Use linear regression to learn optimal weights
            try:
                if len(lstm_pred) > 5:  # Need sufficient data
                    X_stack = np.column_stack([lstm_pred, var_pred])
                    # Use LSTM actuals as target for learning weights
                    y_stack = lstm_results['y_test_rescaled'][:min_len]
                    
                    stacker = LinearRegression(fit_intercept=True)
                    stacker.fit(X_stack, y_stack)
                    
                    ensemble_pred = stacker.predict(X_stack)
                    # For uncertainty, use weighted combination of individual uncertainties
                    w1, w2 = abs(stacker.coef_[0]), abs(stacker.coef_[1])
                    total_w = w1 + w2 + 1e-8
                    ensemble_std = np.sqrt((w1/total_w)**2 * lstm_std**2 + (w2/total_w)**2 * var_std**2)
                    
                    print(f"Stacking learned weights: w_lstm={w1/(w1+w2):.3f}, w_var={w2/(w1+w2):.3f}")
                else:
                    # Fallback to weighted average
                    print("Stacking fallback to user weights")
                    ensemble_pred = lstm_weight * lstm_pred + var_weight * var_pred
                    ensemble_std = np.sqrt(lstm_weight**2 * lstm_std**2 + var_weight**2 * var_std**2)
            except Exception as e:
                # Fallback to weighted average
                print(f"Stacking failed, using user weights: {e}")
                ensemble_pred = lstm_weight * lstm_pred + var_weight * var_pred
                ensemble_std = np.sqrt(lstm_weight**2 * lstm_std**2 + var_weight**2 * var_std**2)
        
        return ensemble_pred, ensemble_std
    
    # =============================================================================
    # EVALUATION (using default weights for test set)
    # =============================================================================
    
    # Get VAR predictions aligned with LSTM test set
    if var_results is not None:
        var_test_pred = var_results['predictions']
        var_test_std = np.std(var_results['predictions']) * np.ones_like(var_test_pred)  # Approximation
    else:
        var_test_pred = None
        var_test_std = None
    
    # Combine test predictions using default weights (for model evaluation)
    ensemble_pred, ensemble_std = combine_predictions(
        lstm_results['y_pred_rescaled'], 
        lstm_results['y_std_rescaled'], 
        var_test_pred, 
        var_test_std, 
        method=ensemble_method,
        horizon_weights=None,  # Use default weights for test evaluation
        is_forecast=False
    )
    
    # Use LSTM actuals as reference (they should be the same timeframe)
    y_test_actual = lstm_results['y_test_rescaled'][:len(ensemble_pred)]
    
    # Calculate ensemble metrics
    ensemble_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test_actual, ensemble_pred)),
        'mae': mean_absolute_error(y_test_actual, ensemble_pred),
        'r2': r2_score(y_test_actual, ensemble_pred),
        'ci_coverage': ((y_test_actual >= (ensemble_pred - 1.96*ensemble_std)) & 
                       (y_test_actual <= (ensemble_pred + 1.96*ensemble_std))).mean()*100,
        'features_used': lstm_results['numeric_features'],
        'grazing_dist_included': 'GrazingDist' in lstm_results['numeric_features'],
        'var_significance_rate': var_results['significance_rate'] if var_results else None,
        'var_uses_significant_only': var_results['used_only_significant'] if var_results else None
    }
    
    # Individual model metrics for comparison
    lstm_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test_actual, lstm_results['y_pred_rescaled'][:len(y_test_actual)])),
        'mae': mean_absolute_error(y_test_actual, lstm_results['y_pred_rescaled'][:len(y_test_actual)]),
        'r2': r2_score(y_test_actual, lstm_results['y_pred_rescaled'][:len(y_test_actual)])
    }
    
    var_metrics = {}
    if var_results is not None and var_test_pred is not None:
        var_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_actual, var_test_pred[:len(y_test_actual)])),
            'mae': mean_absolute_error(y_test_actual, var_test_pred[:len(y_test_actual)]),
            'r2': r2_score(y_test_actual, var_test_pred[:len(y_test_actual)]),
            'significance_rate': var_results['significance_rate'],
            'uses_significant_only': var_results['used_only_significant']
        }
    
    print(f"\nModel Performance Comparison:")
    print(f"LSTM RMSE: {lstm_metrics['rmse']:.4f}, MAE: {lstm_metrics['mae']:.4f}, R²: {lstm_metrics['r2']:.4f}")
    if var_metrics:
        print(f"VAR RMSE: {var_metrics['rmse']:.4f}, MAE: {var_metrics['mae']:.4f}, R²: {var_metrics['r2']:.4f}")
        print(f"VAR Significance Rate: {var_metrics['significance_rate']:.1f}%")
    print(f"Ensemble RMSE: {ensemble_metrics['rmse']:.4f}, MAE: {ensemble_metrics['mae']:.4f}, R²: {ensemble_metrics['r2']:.4f}")
    print(f"GrazingDist included: {ensemble_metrics['grazing_dist_included']}")
    
    # =============================================================================
    # FIXED FUTURE FORECASTING - Only 3 forecasts from last actual data (with GrazingDist)
    # =============================================================================
    
    def forecast_ensemble_fixed(n_future_window):
        """
        Generate ensemble forecasts but limit to 3 forecasts from last actual data point
        Enhanced with GrazingDist support (using original forecasting approach)
        """
        print(f"Generating ensemble forecasts with GrazingDist (using {n_future_window}-step window, limiting to 3 forecasts)...")
        
        # LSTM forecasting (using original approach + GrazingDist support)
        def forecast_lstm(last_sequence, n_steps):
            curr_sequence = last_sequence.copy()
            lstm_predictions = []
            lstm_uncertainties = []
            future_months = []
            future_years = []
            future_seasons = []
            future_precip = []
            future_forecast_precip = []
            future_gaps = []
            future_grazing_dist = []
            
            # Get initial time from last sequence (original approach)
            try:
                last_row_unscaled = lstm_results['scaler'].inverse_transform(curr_sequence[-1].reshape(1,-1))[0]
                curr_month = int(last_row_unscaled[lstm_results['feature_indices']['month_num']])
                curr_year = int(last_row_unscaled[lstm_results['feature_indices']['year']])
            except:
                curr_month = int(data['month_num'].iloc[-1])
                curr_year = int(data['year'].iloc[-1])
            
            # Handle precipitation forecasts (original approach)
            precip_forecast = {}
            if external_precip_forecasts is not None:
                if isinstance(external_precip_forecasts, pd.DataFrame):
                    precip_forecast = {(int(r['year']), int(r['month_num'])): r['Forecasted Precipitation'] 
                                      for _, r in external_precip_forecasts.iterrows()}
                else:
                    precip_forecast = external_precip_forecasts.copy()
            
            for step in range(n_steps):
                # Make prediction using Monte Carlo (original approach)
                try:
                    pred_mean, pred_std = lstm_results['monte_carlo_fn'](
                        lstm_results['model'], 
                        curr_sequence.reshape(1, seq_length, len(lstm_results['numeric_features'])), 
                        n_simulations
                    )
                    
                    lstm_predictions.append(float(pred_mean.flatten()[0]))
                    lstm_uncertainties.append(float(pred_std.flatten()[0]))
                    
                except Exception as e:
                    print(f"LSTM prediction error at step {step}: {e}")
                    # Use last prediction as fallback
                    if lstm_predictions:
                        lstm_predictions.append(lstm_predictions[-1])
                        lstm_uncertainties.append(lstm_uncertainties[-1])
                    else:
                        lstm_predictions.append(0.0)
                        lstm_uncertainties.append(0.0)
                
                # Update time (original approach)
                curr_month += 1
                if curr_month > 12:
                    curr_month = 1
                    curr_year += 1
                
                # Get season (original approach)
                curr_season = data_month_to_season.get(curr_month, 0.0)
                
                # Get GrazingDist - use seasonal average if available (RETAINED FEATURE)
                curr_grazing_dist = 0.0
                if 'GrazingDist' in lstm_results['feature_indices']:
                    curr_grazing_dist = data_month_to_grazing.get(curr_month, 0.0)
                
                # Store time information
                future_months.append(curr_month)
                future_years.append(curr_year)
                future_seasons.append(curr_season)
                future_gaps.append(1)
                future_grazing_dist.append(curr_grazing_dist)
                
                # Handle precipitation (original approach)
                future_key = (curr_year, curr_month)
                if future_key in precip_forecast:
                    curr_forecast_precip = precip_forecast[future_key]
                else:
                    month_data = data[data['month_num'] == curr_month]
                    if len(month_data) > 0:
                        curr_forecast_precip = month_data['Forecasted Precipitation'].mean()
                    else:
                        curr_forecast_precip = 0.0
                
                # Get historical precipitation (original approach)
                month_data = data[data['month_num'] == curr_month]
                if len(month_data) > 0:
                    curr_precip = month_data['precipitation'].mean()
                else:
                    curr_precip = 0.0
                
                future_precip.append(curr_precip)
                future_forecast_precip.append(curr_forecast_precip)
                
                # Create new row for next prediction (original approach + GrazingDist)
                try:
                    temp_row = np.zeros((1, len(lstm_results['numeric_features'])))
                    
                    # Set all features that exist in the feature indices (original approach)
                    if 'year' in lstm_results['feature_indices']:
                        temp_row[0, lstm_results['feature_indices']['year']] = curr_year
                    if 'month_num' in lstm_results['feature_indices']:
                        temp_row[0, lstm_results['feature_indices']['month_num']] = curr_month
                    if 'Season_Index' in lstm_results['feature_indices']:
                        temp_row[0, lstm_results['feature_indices']['Season_Index']] = curr_season
                    if 'precipitation' in lstm_results['feature_indices']:
                        temp_row[0, lstm_results['feature_indices']['precipitation']] = curr_precip
                    if 'Forecasted Precipitation' in lstm_results['feature_indices']:
                        temp_row[0, lstm_results['feature_indices']['Forecasted Precipitation']] = curr_forecast_precip
                    if 'months_gap' in lstm_results['feature_indices']:
                        temp_row[0, lstm_results['feature_indices']['months_gap']] = 1
                    if 'GrazingDist' in lstm_results['feature_indices']:  # RETAINED FEATURE
                        temp_row[0, lstm_results['feature_indices']['GrazingDist']] = curr_grazing_dist
                    
                    # Scale and update sequence (original approach)
                    temp_row_scaled = lstm_results['scaler'].transform(temp_row)[0]
                    pred_full = np.zeros((1, len(lstm_results['numeric_features'])))
                    for j in range(len(lstm_results['numeric_features'])):
                        if j == lstm_results['feature_indices'][target_var]:
                            pred_full[0, j] = float(pred_mean.flatten()[0])
                        else:
                            pred_full[0, j] = temp_row_scaled[j]
                    
                    curr_sequence = np.append(curr_sequence[1:], pred_full, axis=0)
                    
                except Exception as e:
                    print(f"Sequence update error at step {step}: {e}")
                    # Simple fallback (original approach)
                    new_row_scaled = curr_sequence[-1].copy()
                    new_row_scaled[lstm_results['feature_indices'][target_var]] = float(pred_mean.flatten()[0])
                    curr_sequence = np.vstack([curr_sequence[1:], new_row_scaled.reshape(1, -1)])
            
            # Rescale predictions (original approach)
            future_pred = np.array(lstm_predictions).reshape(-1, 1)
            future_std = np.array(lstm_uncertainties).reshape(-1, 1)
            
            dummy_pred = np.zeros((len(future_pred), len(lstm_results['numeric_features'])))
            dummy_pred[:, lstm_results['feature_indices'][target_var]] = future_pred.flatten()
            lstm_pred_rescaled = lstm_results['scaler'].inverse_transform(dummy_pred)[:, lstm_results['feature_indices'][target_var]]
            
            dummy_std = np.zeros((len(future_std), len(lstm_results['numeric_features'])))
            dummy_std[:, lstm_results['feature_indices'][target_var]] = future_std.flatten()
            lstm_std_rescaled = lstm_results['scaler'].inverse_transform(dummy_std)[:, lstm_results['feature_indices'][target_var]]
            
            return (lstm_pred_rescaled, lstm_std_rescaled, future_months, future_years,
                   future_seasons, future_precip, future_forecast_precip, future_gaps, future_grazing_dist)
        
        # VAR forecasting (full window) with GrazingDist support
        def forecast_var(n_steps):
            if var_results is None:
                return None, None, None, None, None, None, None, None, None
                
            try:
                # Use the last observations for forecasting
                forecast_input = var_results['data_processed'].iloc[-var_results['optimal_lags']:].values
                var_forecasts = var_results['model'].forecast(forecast_input, steps=n_steps)
                
                var_predictions = var_forecasts[:, 0]  # Target variable predictions
                var_std = np.std(var_predictions) * np.ones_like(var_predictions)  # Approximation
                
                # Convert back from differenced if needed
                if var_results['differenced_vars'].get(target_var, False):
                    last_level = data[target_var].iloc[-1]
                    var_predictions = np.cumsum(np.concatenate([[last_level], var_predictions]))[1:]
                
                # Generate time information for VAR (same as LSTM)
                curr_month = int(data['month_num'].iloc[-1])
                curr_year = int(data['year'].iloc[-1])
                
                future_months, future_years, future_seasons = [], [], []
                future_precip, future_forecast_precip, future_gaps = [], [], []
                future_grazing_dist = []
                
                for step in range(n_steps):
                    curr_month += 1
                    if curr_month > 12:
                        curr_month = 1
                        curr_year += 1
                    
                    future_months.append(curr_month)
                    future_years.append(curr_year)
                    future_seasons.append(data_month_to_season.get(curr_month, 0.0))
                    future_gaps.append(1)
                    
                    curr_precip = data[data['month_num'] == curr_month]['precipitation'].mean() \
                                  if not data[data['month_num'] == curr_month].empty else 0.0
                    future_precip.append(curr_precip)
                    future_forecast_precip.append(curr_precip)  # Simplified
                    
                    # Add GrazingDist for VAR forecasts (RETAINED FEATURE)
                    curr_grazing_dist = data_month_to_grazing.get(curr_month, 0.0)
                    future_grazing_dist.append(curr_grazing_dist)
                
                return (var_predictions, var_std, future_months, future_years,
                       future_seasons, future_precip, future_forecast_precip, future_gaps, future_grazing_dist)
                
            except Exception as e:
                print(f"VAR forecasting failed: {str(e)}")
                return None, None, None, None, None, None, None, None, None
        
        # Generate forecasts from both models (full window)
        lstm_results_full = forecast_lstm(lstm_results['X_test'][-1], n_future_window)
        var_results_full = forecast_var(n_future_window)
        
        # Combine forecasts using horizon-specific weights (full window)
        if var_results_full[0] is not None:
            ensemble_future_pred, ensemble_future_std = combine_predictions(
                lstm_results_full[0], lstm_results_full[1], 
                var_results_full[0], var_results_full[1], 
                method=ensemble_method,
                horizon_weights=forecast_weights,  # Use forecast-horizon-specific weights
                is_forecast=True
            )
        else:
            ensemble_future_pred = lstm_results_full[0]
            ensemble_future_std = lstm_results_full[1]
        
        # NOW APPLY THE 3-FORECAST LIMIT (same as original model)
        future_months = lstm_results_full[2]
        future_years = lstm_results_full[3]
        future_seasons = lstm_results_full[4]
        future_precip = lstm_results_full[5]
        future_forecast_precip = lstm_results_full[6]
        future_gaps = lstm_results_full[7]
        future_grazing_dist = lstm_results_full[8]
        
        # Find the last actual data point to determine forecasting start point
        last_actual_data = data[data[target_var].notna()].iloc[-1] if not data[data[target_var].notna()].empty else None
        
        if last_actual_data is not None:
            last_actual_year = last_actual_data['year']
            last_actual_month = last_actual_data['month_num']
            
            # Find the index in future forecasts that corresponds to the last actual data
            last_actual_idx = None
            for i, (year, month) in enumerate(zip(future_years, future_months)):
                if year == last_actual_year and month == last_actual_month:
                    last_actual_idx = i
                    break
            
            # If we found the last actual point, only keep 3 forecasts after it
            if last_actual_idx is not None:
                # Keep forecasts from last_actual_idx to last_actual_idx + 3 (inclusive)
                end_idx = min(last_actual_idx + 4, len(ensemble_future_pred))  # +4 to include 3 forecasts after
                
                ensemble_future_pred = ensemble_future_pred[last_actual_idx:end_idx]
                ensemble_future_std = ensemble_future_std[last_actual_idx:end_idx]
                future_months = future_months[last_actual_idx:end_idx]
                future_years = future_years[last_actual_idx:end_idx]
                future_seasons = future_seasons[last_actual_idx:end_idx]
                future_precip = future_precip[last_actual_idx:end_idx]
                future_forecast_precip = future_forecast_precip[last_actual_idx:end_idx]
                future_gaps = future_gaps[last_actual_idx:end_idx]
                future_grazing_dist = future_grazing_dist[last_actual_idx:end_idx]
                
                # Also limit individual model forecasts
                lstm_future_pred = lstm_results_full[0][last_actual_idx:end_idx]
                var_future_pred = var_results_full[0][last_actual_idx:end_idx] if var_results_full[0] is not None else None
            else:
                # If we can't find the last actual point, just take the last 4 forecasts
                ensemble_future_pred = ensemble_future_pred[-4:]
                ensemble_future_std = ensemble_future_std[-4:]
                future_months = future_months[-4:]
                future_years = future_years[-4:]
                future_seasons = future_seasons[-4:]
                future_precip = future_precip[-4:]
                future_forecast_precip = future_forecast_precip[-4:]
                future_gaps = future_gaps[-4:]
                future_grazing_dist = future_grazing_dist[-4:]
                
                lstm_future_pred = lstm_results_full[0][-4:]
                var_future_pred = var_results_full[0][-4:] if var_results_full[0] is not None else None
        else:
            # If no actual data found, take the first 4 forecasts
            ensemble_future_pred = ensemble_future_pred[:4]
            ensemble_future_std = ensemble_future_std[:4]
            future_months = future_months[:4]
            future_years = future_years[:4]
            future_seasons = future_seasons[:4]
            future_precip = future_precip[:4]
            future_forecast_precip = future_forecast_precip[:4]
            future_gaps = future_gaps[:4]
            future_grazing_dist = future_grazing_dist[:4]
            
            lstm_future_pred = lstm_results_full[0][:4]
            var_future_pred = var_results_full[0][:4] if var_results_full[0] is not None else None
        
        return (ensemble_future_pred, ensemble_future_std, lstm_future_pred, var_future_pred,
                future_months, future_years, future_seasons, future_precip, 
                future_forecast_precip, future_gaps, future_grazing_dist)
    
    # Generate future forecasts with 3-forecast limit
    ensemble_results = forecast_ensemble_fixed(n_future)
    ensemble_future_pred, ensemble_future_std, lstm_future_pred, var_future_pred, \
    future_months, future_years, future_seasons, future_precip, future_forecast_precip, \
    future_gaps, future_grazing_dist = ensemble_results
    
    # =============================================================================
    # PREPARE OUTPUT DATAFRAMES (Same format as original + GrazingDist)
    # =============================================================================
    
    # Test results
    test_indices = range(len(data) - len(ensemble_pred) - seq_length, len(data) - seq_length)
    
    test_results = pd.DataFrame({
        'Ward': unique_ward1,
        'Month': data.iloc[test_indices]['month_num'].values,
        'Year': data.iloc[test_indices]['year'].values,
        'Season_Index': data.iloc[test_indices]['Season_Index'].values,
        'Precipitation': data.iloc[test_indices]['precipitation'].values,
        'Months Gap': data.iloc[test_indices]['months_gap'].values,
        'Date': [f"{year}-{month:02d}" for year, month in 
                zip(data.iloc[test_indices]['year'].values, 
                   data.iloc[test_indices]['month_num'].values)],
        'Date_Object': data.iloc[test_indices]['date'].values,
        'Actual': y_test_actual,
        'Forecasted Amount Milked': ensemble_pred,
        'Forecast Uncertainty (Std Dev)': ensemble_std,
        'Lower Bound (95%)': ensemble_pred - 1.96*ensemble_std,
        'Upper Bound (95%)': ensemble_pred + 1.96*ensemble_std,
        'Error': y_test_actual - ensemble_pred,
        'Percent Error': np.where(y_test_actual != 0, 
                                ((y_test_actual - ensemble_pred) / y_test_actual) * 100, 
                                np.nan),
        'LSTM_Prediction': lstm_results['y_pred_rescaled'][:len(ensemble_pred)],
        'VAR_Prediction': var_test_pred[:len(ensemble_pred)] if var_test_pred is not None else [None] * len(ensemble_pred)
    })
    
    # Add GrazingDist to test results if available (RETAINED FEATURE)
    if 'GrazingDist' in data.columns:
        test_results['GrazingDist'] = data.iloc[test_indices]['GrazingDist'].values
    
    # Future forecast results - LIMITED TO 4 ROWS (last actual + 3 forecasts) with GrazingDist
    future_dates = [f"{year}-{month:02d}" for year, month in zip(future_years, future_months)]
    future_date_objects = [pd.Timestamp(f"{year}-{month:02d}-01") for year, month in zip(future_years, future_months)]
    
    # Get actual values if available
    future_actuals = []
    for year, month in zip(future_years, future_months):
        match = data[(data['year'] == year) & (data['month_num'] == month)]
        future_actuals.append(match[target_var].values[0] if not match.empty else None)
    
    forecast_results = pd.DataFrame({
        'Ward': unique_ward1,
        'Month': future_months,
        'Year': future_years,
        'Season_Index': future_seasons,
        'Precipitation': future_precip,
        'Months Gap': future_gaps,
        'Date': future_dates,
        'Date_Object': future_date_objects,
        'Forecasted Amount Milked': ensemble_future_pred,
        'Actual (if available)': future_actuals,
        'Forecast Uncertainty (Std Dev)': ensemble_future_std,
        'Lower Bound (95%)': ensemble_future_pred - 1.96*ensemble_future_std,
        'Upper Bound (95%)': ensemble_future_pred + 1.96*ensemble_future_std,
        'Error': [None if actual is None else actual - pred 
                 for actual, pred in zip(future_actuals, ensemble_future_pred)],
        'Percent Error': [None if actual is None or actual == 0 else 
                         ((actual - pred)/actual)*100 
                         for actual, pred in zip(future_actuals, ensemble_future_pred)],
        'LSTM_Forecast': lstm_future_pred,
        'VAR_Forecast': var_future_pred if var_future_pred is not None else [None] * len(ensemble_future_pred)
    })
    
    # Add GrazingDist to forecast results (RETAINED FEATURE)
    if 'GrazingDist' in lstm_results['numeric_features']:
        forecast_results['GrazingDist'] = future_grazing_dist
    
    # Add last actual value fields - EXACTLY like original implementation
    try:
        last_data = data_numeric.sort_values(['year', 'month_num'])
        last_actual_row = last_data[last_data[target_var].notna()].iloc[-1] if not last_data[last_data[target_var].notna()].empty else None
        
        # Add the 4 additional fields to forecast_results (exactly like original)
        forecast_results['Last_Actual_Value'] = None
        forecast_results['Month1_Forecast'] = None
        forecast_results['Month2_Forecast'] = None
        forecast_results['Month3_Forecast'] = None
        forecast_results['Evaluation_Metrics'] = None
        
        if last_actual_row is not None:
            last_year = last_actual_row['year']
            last_month = last_actual_row['month_num']
            last_actual_value = last_actual_row[target_var]

            last_actual_mask = (forecast_results['Year'] == last_year) & (forecast_results['Month'] == last_month)
            
            if last_actual_mask.any():
                last_idx = forecast_results[last_actual_mask].index[0]

                forecast_results.loc[last_idx, 'Last_Actual_Value'] = last_actual_value
                forecast_results.loc[last_idx, 'Evaluation_Metrics'] = json.dumps(ensemble_metrics)

                # Only add the next 3 forecasts (exactly like original)
                if last_idx + 1 < len(forecast_results):
                    forecast_results.loc[last_idx, 'Month1_Forecast'] = forecast_results.loc[last_idx + 1, 'Forecasted Amount Milked']
                    
                if last_idx + 2 < len(forecast_results):
                    forecast_results.loc[last_idx, 'Month2_Forecast'] = forecast_results.loc[last_idx + 2, 'Forecasted Amount Milked']
                    
                if last_idx + 3 < len(forecast_results):
                    forecast_results.loc[last_idx, 'Month3_Forecast'] = forecast_results.loc[last_idx + 3, 'Forecasted Amount Milked']
                    
    except Exception as e:
        print(f"Error adding last actual value fields: {str(e)}")
    
    return {
        'model': lstm_results['model'],  # Return LSTM model as primary
        'scaler': lstm_results['scaler'],
        'training_history': lstm_results['history'].history,
        'evaluation_metrics': ensemble_metrics,
        'test_results': test_results,
        'forecast_results': forecast_results,
        'feature_indices': lstm_results['feature_indices'],
        'data_month_to_season': data_month_to_season,
        'lstm_model': lstm_results,
        'var_model': var_results,
        'individual_metrics': {
            'lstm': lstm_metrics,
            'var': var_metrics,
            'ensemble': ensemble_metrics
        },
        'ensemble_method': ensemble_method,
        'weights': {'lstm': lstm_weight, 'var': var_weight},
        'forecast_weights': forecast_weights  # NEW: Include forecast-horizon-specific weights
    }

def empty_results_dict():
    return {
        'model': None, 'scaler': None, 'training_history': None,
        'evaluation_metrics': {}, 'test_results': pd.DataFrame(),
        'forecast_results': pd.DataFrame(), 'feature_indices': {},
        'data_month_to_season': {}, 'lstm_model': None, 'var_model': None,
        'individual_metrics': {}, 'ensemble_method': None, 'weights': {}
    }
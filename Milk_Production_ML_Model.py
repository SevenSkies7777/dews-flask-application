#importing libraries
import pandas as pd
import mysql.connector

conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Romans17:48',
        database='livelihoodzones'
    )

cursor = conn.cursor()

query = """
    SELECT hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId as qid, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.ExerciseStartDate, Sum(hh_livestock_milk_production_per_species.DailyQntyMilkedInLtrs) as amountmilked,Sum(hh_livestock_milk_production_per_species.DailyQntyConsumedInLtrs) as amountconsumed,Sum(hh_livestock_milk_production_per_species.DailyQntySoldInLtrs) as amountsold, Sum(hh_livestock_milk_production_per_species.PricePerLtr) as PricePerLtr,wards.Shapefile_wardName
    FROM (hh_livestock_milk_production_per_species
          LEFT JOIN hha_questionnaire_sessions ON (hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)LEFT JOIN wards ON (hha_questionnaire_sessions.WardId = wards.WardId)
    WHERE (hha_questionnaire_sessions.CountyId = '46' )
    GROUP BY hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,data_collection_exercise.ExerciseStartDate, wards.Shapefile_wardName
"""

db_df1 = pd.read_sql(query, conn)

query = """
    SELECT Seasons.season,Seasons.Season_Index, Seasons.Month,LTAs.Bad_year, LTAs.Good_year
    FROM Seasons LEFT JOIN LTAs ON (Seasons.month = LTAs.month)
   WHERE (LTAs.CountyId = '46')
    """

Seasons = pd.read_sql(query, conn)

db_df1['year'] = db_df1['ExerciseStartDate'].dt.year
db_df1['month'] = db_df1['ExerciseStartDate'].dt.strftime('%B') 
db_df1['month_num'] = db_df1['ExerciseStartDate'].dt.month

db_df = db_df1.merge(Seasons, left_on=['month'], right_on=['Month'], how='right')

conn.close()
#db_df

conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Romans17:48',
        database='livelihoodzones'
    )

cursor = conn.cursor()

query = """
    SELECT *
    FROM Precipitation LEFT JOIN counties ON (counties.CountyName = Precipitation.NAME_1)
    WHERE (counties.CountyId = '46')
    """

precipitation_df = pd.read_sql(query, conn)

prep_df0 = precipitation_df.groupby(['NAME_3','T'])['precipitation'].sum()
prep_df0 = prep_df0.reset_index()

conn.close()
prep_df0['T'] = pd.to_datetime(prep_df0['T'], errors='coerce')
prep_df0['year'] = prep_df0['T'].dt.year
prep_df0['month_name'] = prep_df0['T'].dt.strftime('%B') 
prep_df0['month_num'] = prep_df0['T'].dt.month
prep_df0
prep_df0 = Seasons.merge(prep_df0, left_on=['Month'], right_on=['month_name'], how='right')
#prep_df0

#unique_wards = prep_df0["NAME_3"].unique()
unique_wards = ['Wumingu/Kishushe','Wusi/Kishamba']
for NAME_3 in unique_wards:
    print(f"Processing {NAME_3}...")
    prep_df = prep_df0[prep_df0["NAME_3"] == NAME_3]
    prep_df = prep_df.reset_index()    
    prep_df=prep_df[['season','Season_Index','Month','NAME_3','T','precipitation','year','month_name','month_num']]
    unique_ward = prep_df["NAME_3"].unique()
    
#LSTM MOdel Precipitation
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define function for Monte Carlo Dropout
def monte_carlo_predictions(model, X_input, n_simulations=100):
    @tf.function
    def f_model_mc(X_input, training=True):
        return model(X_input, training=training)
    
    preds = np.array([f_model_mc(X_input, training=True).numpy() for _ in range(n_simulations)])
    return preds.mean(axis=0), preds.std(axis=0)  # Mean and standard deviation of predictions

data = prep_df.copy()  # Replace with actual data

data = data.sort_values(['year', 'month_num'])

data['month_year'] = data['year'] * 12 + data['month_num']
data['months_gap'] = data['month_year'].diff().fillna(1).astype(int)

features_precip = ["year", "month_num", "Season_Index", "precipitation", "months_gap"]
data_precip = data[features_precip]

year_idx = 0
month_idx = 1
season_idx = 2
target_var_idx = 3  # "precipitation" is the 4th column (index 3)
gap_idx = 4  # "months_gap" is the 5th column (index 4)

data_month_to_precip = {}
for month in range(1, 13):
    month_data = data[data['month_num'] == month]
    if len(month_data) > 0:
        data_month_to_precip[month] = month_data['precipitation'].mean()

data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month_num'].astype(str).str.zfill(2) + '-01')

plt.figure(figsize=(12, 6))

plt.plot(data['date'], data['precipitation'], 'b-o', markersize=5, alpha=0.7)

gap_points = data[data['months_gap'] > 1]
if len(gap_points) > 0:
    plt.scatter(gap_points['date'], 
                gap_points['precipitation'], 
                c='red', s=80, zorder=5, label=f'Gaps (>{gap_points["months_gap"].min()} month)')

plt.title('Precipitation Time Series with Gaps Highlighted')
plt.ylabel('Precipitation')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

gap_counts = data['months_gap'].value_counts().sort_index()
print("Gap distribution (months between observations):")
print(gap_counts)
print(f"Maximum gap: {data['months_gap'].max()} months")
print(f"Total observations: {len(data)}")
print(f"Observations with gaps > 1 month: {len(gap_points)} ({len(gap_points)/len(data)*100:.1f}%)")

data_for_scaling = data_precip.copy()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_for_scaling)

def preprocess_data(data, target_var_idx, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length, :])
        y.append(data[i + seq_length, target_var_idx])
    return np.array(X), np.array(y)

# Prepare data for precipitation prediction
seq_length = 48  # Use 12 months to predict next month's precipitation
X, y = preprocess_data(data_scaled, target_var_idx, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Build LSTM Model with Dropout for precipitation prediction
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),  # Keep dropout to enable MC Dropout
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(16),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True
)

# Train model
print("Training precipitation prediction model...")
epochs = 70  # Fewer epochs for simpler model
batch_size = 16
history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Precipitation Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Monte Carlo Dropout Predictions
n_simulations = 100  # Number of forward passes
y_pred_mean, y_pred_std = monte_carlo_predictions(model, X_test, n_simulations)

dummy_pred = np.zeros((len(y_pred_mean), data_scaled.shape[1]))
dummy_pred[:, target_var_idx] = y_pred_mean.flatten()
y_pred_mean_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]

dummy_std = np.zeros((len(y_pred_std), data_scaled.shape[1]))
dummy_std[:, target_var_idx] = y_pred_std.flatten()
y_pred_std_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]

dummy_y = np.zeros((len(y_test), data_scaled.shape[1]))
dummy_y[:, target_var_idx] = y_test
y_test_rescaled = scaler.inverse_transform(dummy_y)[:, target_var_idx]

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, 'b-', label='Actual Precipitation')
plt.plot(y_pred_mean_rescaled, 'r-', label='Predicted Precipitation')
plt.fill_between(
    range(len(y_pred_mean_rescaled)),
    y_pred_mean_rescaled - 1.96 * y_pred_std_rescaled,
    y_pred_mean_rescaled + 1.96 * y_pred_std_rescaled,
    color='pink', alpha=0.3, label='95% Confidence Interval'
)
plt.title('Precipitation: Actual vs Predicted with Uncertainty')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

valid_indices = ~np.isnan(y_test_rescaled) & ~np.isnan(y_pred_mean_rescaled)
if not all(valid_indices):
    print(f"Warning: Found {sum(~valid_indices)} NaN values in test data or predictions.")
    y_test_rescaled = y_test_rescaled[valid_indices]
    y_pred_mean_rescaled = y_pred_mean_rescaled[valid_indices]
    y_pred_std_rescaled = y_pred_std_rescaled[valid_indices]

mse = mean_squared_error(y_test_rescaled, y_pred_mean_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_mean_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_mean_rescaled)

print(f"Precipitation Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

start_idx = len(data_for_scaling) - len(y_test_rescaled) - seq_length
end_idx = len(data_for_scaling) - seq_length
test_indices = list(range(start_idx, end_idx))

test_months = data.iloc[test_indices]["month_num"].values
test_years = data.iloc[test_indices]["year"].values
test_seasons = data.iloc[test_indices]["Season_Index"].values
test_gaps = data.iloc[test_indices]["months_gap"].values
test_dates = [f"{year}-{month:02d}" for year, month in zip(test_years, test_months)]
test_date_objects = data.iloc[test_indices]["date"].values

df_predictions = pd.DataFrame({
    "Month": test_months,
    "Year": test_years,
    "Season_Index": test_seasons,
    "Months Gap": test_gaps,
    "Date": test_dates,
    "Actual Precipitation": y_test_rescaled,
    "Forecasted Precipitation": y_pred_mean_rescaled,
    "Forecast Uncertainty (Std Dev)": y_pred_std_rescaled,
    "Lower Bound (95%)": y_pred_mean_rescaled - 1.96 * y_pred_std_rescaled,
    "Upper Bound (95%)": y_pred_mean_rescaled + 1.96 * y_pred_std_rescaled,
    "Error": y_test_rescaled - y_pred_mean_rescaled,
    "Percent Error": ((y_test_rescaled - y_pred_mean_rescaled) / y_test_rescaled) * 100
})

within_ci = ((df_predictions["Actual Precipitation"] >= df_predictions["Lower Bound (95%)"]) & 
             (df_predictions["Actual Precipitation"] <= df_predictions["Upper Bound (95%)"]))
ci_coverage = within_ci.mean() * 100

print(f"Percentage of actual values within 95% confidence interval: {ci_coverage:.1f}%")

def forecast_future_precipitation(model, last_sequence, n_future, scaler, target_var_idx, 
                                 year_idx, month_idx, season_idx, gap_idx, 
                                 month_to_precip, n_simulations=100):
    """
    Forecast n_future time steps ahead using the trained model, explicitly updating
    month, year, season and gap features for each step
    """
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
    curr_month = int(last_row_unscaled[month_idx])
    curr_year = int(last_row_unscaled[year_idx])
    
    for i in range(n_future):
        # Get prediction with uncertainty
        pred_mean, pred_std = monte_carlo_predictions(model, curr_sequence.reshape(1, seq_length, n_features), n_simulations)

        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1

        months_gap = 1

        matching_rows = data[data['month_num'] == curr_month]
        if len(matching_rows) > 0:
            curr_season = matching_rows['Season_Index'].mean()
        else:
            curr_season = 0.0

        future_months.append(curr_month)
        future_years.append(curr_year)
        future_seasons.append(curr_season)
        future_gaps.append(months_gap)

        temp_row = np.zeros((1, n_features))
        temp_row[0, year_idx] = curr_year
        temp_row[0, month_idx] = curr_month
        temp_row[0, season_idx] = curr_season
        temp_row[0, gap_idx] = months_gap

        temp_row_scaled = scaler.transform(temp_row)[0]

        pred_full = np.zeros((1, n_features))
        for j in range(n_features):
            if j == target_var_idx:
                pred_full[0, j] = pred_mean[0, 0]  # Keep the predicted precipitation value
            else:
                pred_full[0, j] = temp_row_scaled[j]

        curr_sequence = np.append(curr_sequence[1:], pred_full, axis=0)

        future_predictions.append(pred_mean[0, 0])
        prediction_std.append(pred_std[0, 0])

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    prediction_std = np.array(prediction_std).reshape(-1, 1)

    dummy_pred = np.zeros((len(future_predictions), n_features))
    dummy_pred[:, target_var_idx] = future_predictions.flatten()
    future_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]
    
    dummy_std = np.zeros((len(prediction_std), n_features))
    dummy_std[:, target_var_idx] = prediction_std.flatten()
    future_std_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]
    
    return future_pred_rescaled, future_std_rescaled, future_months, future_years, future_seasons, future_gaps

forecast_start_year = 2016
forecast_start_month = 1

target_date = pd.Timestamp(f"{forecast_start_year}-{forecast_start_month:02d}-01")
closest_idx = (data['date'] - target_date).abs().idxmin()

if closest_idx < seq_length:
    print(f"Warning: Not enough history before {target_date}. Using earliest possible sequence.")
    start_seq_idx = 0
else:
    start_seq_idx = closest_idx - seq_length

forecast_start_sequence = data_scaled[start_seq_idx:start_seq_idx+seq_length]

# Number of months to forecast
n_future = 109  # Forecasting 3 years ahead from Jan 2016

print(f"Starting forecast from {data.loc[closest_idx, 'date'].strftime('%Y-%m')} with {n_future} months ahead")
future_pred, future_std, future_months, future_years, future_seasons, future_gaps = forecast_future_precipitation(
    model, forecast_start_sequence, n_future, scaler, target_var_idx, year_idx, month_idx, season_idx, gap_idx,
    data_month_to_precip, n_simulations=100
)

future_dates = [f"{year}-{month:02d}" for year, month in zip(future_years, future_months)]
future_date_objects = [pd.Timestamp(f"{year}-{month:02d}-01") for year, month in zip(future_years, future_months)]

future_actuals = []
for year, month in zip(future_years, future_months):
    matching_rows = data[(data["year"] == year) & (data["month_num"] == month)]
    if len(matching_rows) > 0:        
        actual_value = matching_rows["precipitation"].values[0]
        future_actuals.append(actual_value)
    else:
        future_actuals.append(None)

forecast_df = pd.DataFrame({
    'Month': future_months,
    'Year': future_years,
    'Season_Index': future_seasons,
    'Months Gap': future_gaps,
    'Date': future_dates,
    'Date_Object': future_date_objects,
    'Forecasted Precipitation': future_pred,
    'Actual (if available)': future_actuals,
    'Forecast Uncertainty (Std Dev)': future_std,
    'Lower Bound (95%)': future_pred - 1.96 * future_std,
    'Upper Bound (95%)': future_pred + 1.96 * future_std
})

forecast_df['Error'] = None
forecast_df['Percent Error'] = None

for i in range(len(forecast_df)):
    if forecast_df['Actual (if available)'].iloc[i] is not None:
        forecast_df['Error'].iloc[i] = forecast_df['Actual (if available)'].iloc[i] - forecast_df['Forecasted Precipitation'].iloc[i]
        if forecast_df['Actual (if available)'].iloc[i] != 0:  # Avoid division by zero
            forecast_df['Percent Error'].iloc[i] = (forecast_df['Error'].iloc[i] / forecast_df['Actual (if available)'].iloc[i]) * 100

print(f"\nFuture Precipitation Forecasts (starting from {forecast_start_year}-{forecast_start_month:02d}):")
print(forecast_df[['Date', 'Forecasted Precipitation', 'Forecast Uncertainty (Std Dev)', 'Lower Bound (95%)', 'Upper Bound (95%)']].head(20))

forecast_with_actuals = forecast_df.dropna(subset=['Actual (if available)'])
if len(forecast_with_actuals) > 0:
    actual_values = forecast_with_actuals['Actual (if available)'].values
    predicted_values = forecast_with_actuals['Forecasted Precipitation'].values
    
    forecast_mse = mean_squared_error(actual_values, predicted_values)
    forecast_rmse = np.sqrt(forecast_mse)
    forecast_mae = mean_absolute_error(actual_values, predicted_values)
    forecast_r2 = r2_score(actual_values, predicted_values)
    
    print(f"\nForecast Evaluation Metrics (for periods with actual data, n={len(forecast_with_actuals)}):")
    print(f"Root Mean Squared Error (RMSE): {forecast_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {forecast_mae:.4f}")
    print(f"R² Score: {forecast_r2:.4f}")

    within_ci = ((forecast_with_actuals['Actual (if available)'] >= forecast_with_actuals['Lower Bound (95%)']) &
                 (forecast_with_actuals['Actual (if available)'] <= forecast_with_actuals['Upper Bound (95%)']))
    ci_coverage_forecast = within_ci.mean() * 100
    print(f"Percentage of actual values within 95% confidence interval: {ci_coverage_forecast:.1f}%")

plt.figure(figsize=(15, 8))

history_start_idx = max(0, start_seq_idx)
history_end_idx = start_seq_idx + seq_length
history_dates = data.iloc[history_start_idx:history_end_idx]['date'].values
history_values = data.iloc[history_start_idx:history_end_idx]['precipitation'].values

plt.plot(history_dates, history_values, 'b-o', markersize=4, label='Historical Precipitation')

plt.plot(future_date_objects, future_pred, 'r-', label='Forecast')
plt.fill_between(
    future_date_objects,
    future_pred - 1.96 * future_std,
    future_pred + 1.96 * future_std,
    color='pink', alpha=0.3, label='95% Confidence Interval'
)

actuals_dates = []
actuals_values = []
for i, actual in enumerate(future_actuals):
    if actual is not None:
        actuals_dates.append(future_date_objects[i])
        actuals_values.append(actual)

if len(actuals_dates) > 0:
    plt.plot(actuals_dates, actuals_values, 'go', markersize=5, label='Actual Values')

forecast_start_date = future_date_objects[0]
plt.axvline(x=forecast_start_date, color='k', linestyle='--', label='Forecast Start')

plt.title('Precipitation Future Forecast with Uncertainty and Available Actuals')
plt.legend()
plt.ylabel('Precipitation')
plt.xlabel('Date')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Forecasted precipitation (mean):", y_pred_mean_rescaled[:5], "...")  # Show first 5 values
print("Forecast Uncertainty (std dev):", y_pred_std_rescaled[:5], "...")  # Show first 5 values

def predict_all_precipitation(model, data_scaled, seq_length, target_var_idx, scaler, n_simulations=100):
    """
    Generate predictions for all data points that have enough history
    """
    predictions = []
    uncertainty = []
 
    for i in range(seq_length, len(data_scaled)):

        sequence = data_scaled[i-seq_length:i].reshape(1, seq_length, data_scaled.shape[1])
        pred_mean, pred_std = monte_carlo_predictions(model, sequence, n_simulations)
        predictions.append(pred_mean[0, 0])
        uncertainty.append(pred_std[0, 0])
    predictions = np.array(predictions).reshape(-1, 1)
    uncertainty = np.array(uncertainty).reshape(-1, 1)
    dummy_pred = np.zeros((len(predictions), data_scaled.shape[1]))
    dummy_pred[:, target_var_idx] = predictions.flatten()
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]
    
    dummy_std = np.zeros((len(uncertainty), data_scaled.shape[1]))
    dummy_std[:, target_var_idx] = uncertainty.flatten()
    uncertainty_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]
    
    return predictions_rescaled, uncertainty_rescaled

all_predictions, all_uncertainty = predict_all_precipitation(model, data_scaled, seq_length, target_var_idx, scaler)

all_dates = data.iloc[seq_length:]['date'].values
all_actuals = data.iloc[seq_length:]['precipitation'].values

plt.figure(figsize=(15, 8))
plt.plot(all_dates, all_actuals, 'b-', label='Actual Precipitation')
plt.plot(all_dates, all_predictions, 'r-', label='Model Predictions')
plt.fill_between(
    all_dates,
    all_predictions - 1.96 * all_uncertainty,
    all_predictions + 1.96 * all_uncertainty,
    color='pink', alpha=0.3, label='95% Confidence Interval'
)
plt.title('Precipitation: Actual vs Predicted (All Available Data)')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

precipitation_predictions = pd.DataFrame({
    'date': all_dates,
    'actual_precipitation': all_actuals,
    'predicted_precipitation': all_predictions,
    'precipitation_uncertainty': all_uncertainty
})

print("\nPrecipitation model complete. Predictions are ready for use in milk production forecasting.")

#merging Socio-economic data with Rainfall data
forecast_df1=forecast_df[['Month','Year','Date_Object','Forecasted Precipitation','Forecast Uncertainty (Std Dev)']]
prep_df1 = forecast_df1.merge(prep_df, left_on=['Month','Year','Date_Object'], right_on=['month_num','year','T'], how='right')
prep_df2=prep_df1[['year','Forecasted Precipitation','NAME_3','T','precipitation','month_name','month_num']]
precipitation_forecasts_df=prep_df2
db_df=db_df[['WardId','HouseHoldId','Shapefile_wardName', 'month', 'year', 'season','Season_Index','amountmilked','Bad_year','Good_year']]
#db_df=db_df[(db_df['Shapefile_wardName']=="Wusi/Kishamba")&(db_df['year']==2024)]
#db_df

#Cleaning data (Outliers)
def replace_outliers_with_averages(db_df, variables):
    """
    Replace outliers with the average value for each ward and month combination.
    
    Parameters:
    -----------
    db_df : pandas.DataFrame
        DataFrame with outlier detection results
    variables : list
        List of variable names to process
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers replaced by averages
    """
    # Create a clean version with outliers replaced by averages
    db_df_clean = db_df.copy()
    
    # Make sure we have the year_month column (in case it was removed)
    if 'year_month' not in db_df.columns:
        db_df_clean['year_month'] = db_df_clean['year'].astype(str) + '-' + db_df_clean['month'].astype(str)
    
    for variable in variables:
        print(f"\nReplacing outliers for {variable} with ward-month averages")
        outlier_col = f'{variable}_is_outlier'
        
        # Check if outlier column exists
        if outlier_col not in db_df.columns:
            print(f"Warning: No outlier data found for {variable}, skipping replacement")
            continue
        
        # Iterate through each ward and year-month
        for ward in db_df_clean['ward'].unique():
            for year_month in db_df_clean['year_month'].unique():
                # Get data for this ward and this year-month
                mask = (db_df_clean['ward'] == ward) & (db_df_clean['year_month'] == year_month)
                subset = db_df_clean[mask]
                
                # Skip if no data
                if len(subset) == 0:
                    continue
                
                # Skip if insufficient data
                if len(subset) <= 2:
                    print(f"Warning: Ward {ward} in {year_month} has insufficient data for average calculation")
                    continue
                    
                # Check if there are any outliers in this group
                if not subset[outlier_col].any():
                    continue
                
                # Check if there are any non-outliers to calculate average from
                if subset[outlier_col].all():
                    print(f"Warning: All values in ward {ward}, {year_month} for {variable} are outliers")
                    # Try to use the ward average instead
                    ward_avg = db_df_clean.loc[(db_df_clean['ward'] == ward) & ~db_df_clean[outlier_col], variable].mean()
                    if np.isnan(ward_avg):
                        print(f"Cannot find suitable replacement for ward {ward}, {year_month}. Keeping original values.")
                        continue
                    replacement_value = ward_avg
                else:
                    # Calculate mean of non-outlier values
                    non_outlier_mean = subset.loc[~subset[outlier_col], variable].mean()
                    replacement_value = non_outlier_mean
                
                # Replace outliers with the mean
                outlier_mask = mask & db_df_clean[outlier_col]
                if outlier_mask.any():
                    db_df_clean.loc[outlier_mask, variable] = replacement_value
                    print(f"Replaced {outlier_mask.sum()} outliers in ward {ward}, {year_month} for {variable}")
    
    return db_df_clean

# Example usage (to be appended to your code):
# After running analyze_outliers:
db_df_clean = replace_outliers_with_averages(db_df, variables=['amountmilked'])
db_df_clean

unique_ward
unique_ward1 = unique_ward[0]
unique_ward1

db_df_clean1=db_df_clean.groupby(['Shapefile_wardName', 'month', 'year', 'season','Season_Index','Bad_year','Good_year'])[['amountmilked']].mean().reset_index()

joined_data2 = db_df_clean1.merge(prep_df2, left_on=['Shapefile_wardName', 'year', 'month'], right_on=['NAME_3', 'year', 'month_name'], how='right')

joined_data3=joined_data2[(joined_data2['Shapefile_wardName']==unique_ward1)&(joined_data2['year']>2016)]

data_numeric = joined_data3.assign(**{col: joined_data3[col].map(lambda x: x.toordinal()) 
                                      for col in joined_data3.select_dtypes(include=['datetime64'])})

data_numeric
data_numeric = data_numeric.sort_values(by="T")
data_numeric



#LSTM MOdel1
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def monte_carlo_predictions(model, X_input, n_simulations=100):
    @tf.function
    def f_model_mc(X_input, training=True):
        return model(X_input, training=training)
    
    preds = np.array([f_model_mc(X_input, training=True).numpy() for _ in range(n_simulations)])
    return preds.mean(axis=0), preds.std(axis=0)  # Mean and standard deviation of predictions

data = data_numeric.copy()  # Replace with actual data

data = data.sort_values(['year', 'month_num'])

data['month_year'] = data['year'] * 12 + data['month_num']
# Calculate the difference to get the gap
data['months_gap'] = data['month_year'].diff().fillna(1).astype(int)

features = ["year", "month_num", "Season_Index", "precipitation", "Forecasted Precipitation", "amountmilked", "months_gap"]
data = data[features]

year_idx = 0
month_idx = 1
season_idx = 2
precip_idx = 3
forecast_precip_idx = 4
target_var_idx = 5  # "amountmilked" is the 6th column (index 5)
gap_idx = 6  # "months_gap" is the 7th column (index 6)

data_month_to_season = {}
for month in range(1, 13):
    month_data = data[data['month_num'] == month]
    if len(month_data) > 0:
        data_month_to_season[month] = month_data['Season_Index'].mean()

# Create date field for plotting (but keep it separate from the features for scaling)
data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month_num'].astype(str).str.zfill(2) + '-01')

plt.figure(figsize=(12, 6))


plt.plot(data['date'], data['amountmilked'], 'b-o', markersize=5, alpha=0.7)

gap_points = data[data['months_gap'] > 1]
if len(gap_points) > 0:
    plt.scatter(gap_points['date'], 
                gap_points['amountmilked'], 
                c='red', s=80, zorder=5, label=f'Gaps (>{gap_points["months_gap"].min()} month)')

plt.title('Milk Production Time Series with Gaps Highlighted')
plt.ylabel('Amount Milked')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

gap_counts = data['months_gap'].value_counts().sort_index()
print("Gap distribution (months between observations):")
print(gap_counts)
print(f"Maximum gap: {data['months_gap'].max()} months")
print(f"Total observations: {len(data)}")
print(f"Observations with gaps > 1 month: {len(gap_points)} ({len(gap_points)/len(data)*100:.1f}%)")

data_for_scaling = data[features].copy()  # Use only the original numerical features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_for_scaling)

# Prepare sequences function
def preprocess_data(data, target_var_idx, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length, :])
        y.append(data[i + seq_length, target_var_idx])
    return np.array(X), np.array(y)

# Prepare data
seq_length = 13
X, y = preprocess_data(data_scaled, target_var_idx, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)


model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),  # Keep dropout to enable MC Dropout
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True
)

# Train model
epochs = 100  # Increased epochs with early stopping
batch_size = 16
history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

n_simulations = 100  # Number of forward passes
y_pred_mean, y_pred_std = monte_carlo_predictions(model, X_test, n_simulations)

dummy_pred = np.zeros((len(y_pred_mean), data_scaled.shape[1]))
dummy_pred[:, target_var_idx] = y_pred_mean.flatten()
y_pred_mean_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]

dummy_std = np.zeros((len(y_pred_std), data_scaled.shape[1]))
dummy_std[:, target_var_idx] = y_pred_std.flatten()
y_pred_std_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]

dummy_y = np.zeros((len(y_test), data_scaled.shape[1]))
dummy_y[:, target_var_idx] = y_test
y_test_rescaled = scaler.inverse_transform(dummy_y)[:, target_var_idx]

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, 'b-', label='Actual')
plt.plot(y_pred_mean_rescaled, 'r-', label='Predicted')
plt.fill_between(
    range(len(y_pred_mean_rescaled)),
    y_pred_mean_rescaled - 1.96 * y_pred_std_rescaled,
    y_pred_mean_rescaled + 1.96 * y_pred_std_rescaled,
    color='pink', alpha=0.3, label='95% Confidence Interval'
)
plt.title('Milk Production: Actual vs Predicted with Uncertainty')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amount Milked')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

valid_indices = ~np.isnan(y_test_rescaled) & ~np.isnan(y_pred_mean_rescaled)
if not all(valid_indices):
    print(f"Warning: Found {sum(~valid_indices)} NaN values in test data or predictions.")
    y_test_rescaled = y_test_rescaled[valid_indices]
    y_pred_mean_rescaled = y_pred_mean_rescaled[valid_indices]
    y_pred_std_rescaled = y_pred_std_rescaled[valid_indices]


mse = mean_squared_error(y_test_rescaled, y_pred_mean_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_mean_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_mean_rescaled)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")


start_idx = len(data_for_scaling) - len(y_test_rescaled) - seq_length
end_idx = len(data_for_scaling) - seq_length
test_indices = list(range(start_idx, end_idx))


test_months = data.iloc[test_indices]["month_num"].values
test_years = data.iloc[test_indices]["year"].values
test_seasons = data.iloc[test_indices]["Season_Index"].values
test_precip = data.iloc[test_indices]["precipitation"].values
test_forecast_precip = data.iloc[test_indices]["Forecasted Precipitation"].values
test_gaps = data.iloc[test_indices]["months_gap"].values
test_dates = [f"{year}-{month:02d}" for year, month in zip(test_years, test_months)]
test_date_objects = data.iloc[test_indices]["date"].values


df_predictions = pd.DataFrame({
    "Month": test_months,
    "Year": test_years,
    "Season_Index": test_seasons,
    "Precipitation": test_precip,
    "Months Gap": test_gaps,
    "Date": test_dates,
    "Actual": y_test_rescaled,
    "Forecasted Amount Milked": y_pred_mean_rescaled,
    "Forecast Uncertainty (Std Dev)": y_pred_std_rescaled,
    "Lower Bound (95%)": y_pred_mean_rescaled - 1.96 * y_pred_std_rescaled,
    "Upper Bound (95%)": y_pred_mean_rescaled + 1.96 * y_pred_std_rescaled,
    "Error": y_test_rescaled - y_pred_mean_rescaled,
    "Percent Error": ((y_test_rescaled - y_pred_mean_rescaled) / y_test_rescaled) * 100
})


within_ci = ((df_predictions["Actual"] >= df_predictions["Lower Bound (95%)"]) & 
             (df_predictions["Actual"] <= df_predictions["Upper Bound (95%)"]))
ci_coverage = within_ci.mean() * 100

print(f"Percentage of actual values within 95% confidence interval: {ci_coverage:.1f}%")


def forecast_future_with_gap_tracking(model, last_sequence, n_future, scaler, target_var_idx, 
                                     year_idx, month_idx, season_idx, precip_idx, forecast_precip_idx,
                                     gap_idx, month_to_season, future_precip_forecast=None, n_simulations=100):
    """
    Forecast n_future time steps ahead using the trained model, explicitly updating
    month, year, season, forecasted precipitation, and gap features for each step
    
    Parameters:
    -----------
    model : Keras model
        The trained LSTM model for prediction
    last_sequence : numpy array
        The last sequence of data used as the starting point
    n_future : int
        Number of future time steps to predict
    scaler : MinMaxScaler
        The scaler used to normalize the data
    target_var_idx : int
        Index of the target variable (milk production)
    year_idx, month_idx, season_idx, precip_idx, forecast_precip_idx, gap_idx : int
        Indices for the respective features
    month_to_season : dict
        Dictionary mapping month to average season index
    future_precip_forecast : dict, optional
        Dictionary mapping (year, month) tuples to forecasted precipitation values
    n_simulations : int, default=100
        Number of Monte Carlo simulations for uncertainty estimation
    """
    future_predictions = []
    prediction_std = []
    future_months = []
    future_years = []
    future_seasons = []
    future_precip = []
    future_forecast_precip = []
    future_gaps = []
    

    curr_sequence = last_sequence.copy()
    

    seq_length = curr_sequence.shape[0]
    n_features = curr_sequence.shape[1]
    

    last_row_unscaled = scaler.inverse_transform(curr_sequence[-1].reshape(1, -1))[0]
    curr_month = int(last_row_unscaled[month_idx])
    curr_year = int(last_row_unscaled[year_idx])
    

    if future_precip_forecast is None:
        future_precip_forecast = {}
    
    for i in range(n_future):

        pred_mean, pred_std = monte_carlo_predictions(model, curr_sequence.reshape(1, seq_length, n_features), n_simulations)
        

        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1
        

        months_gap = 1

        if curr_month in month_to_season:
            curr_season = month_to_season[curr_month]
        else:

            curr_season = 0.0
        

        future_key = (curr_year, curr_month)
        if future_key in future_precip_forecast:
            
            curr_forecast_precip = future_precip_forecast[future_key]
        else:
            
            month_data = data[data['month_num'] == curr_month]
            if len(month_data) > 0:
                curr_forecast_precip = month_data['Forecasted Precipitation'].mean()
            else:
                curr_forecast_precip = 0.0
        

        month_data = data[data['month_num'] == curr_month]
        if len(month_data) > 0:
            curr_precip = month_data['precipitation'].mean()
        else:
            curr_precip = 0.0
        

        future_months.append(curr_month)
        future_years.append(curr_year)
        future_seasons.append(curr_season)
        future_precip.append(curr_precip)
        future_forecast_precip.append(curr_forecast_precip)
        future_gaps.append(months_gap)
        

        temp_row = np.zeros((1, n_features))
        temp_row[0, year_idx] = curr_year
        temp_row[0, month_idx] = curr_month
        temp_row[0, season_idx] = curr_season
        temp_row[0, precip_idx] = curr_precip
        temp_row[0, forecast_precip_idx] = curr_forecast_precip
        temp_row[0, gap_idx] = months_gap
        
        
        temp_row_scaled = scaler.transform(temp_row)[0]
        
        
        pred_full = np.zeros((1, n_features))
        for j in range(n_features):
            if j == target_var_idx:
                pred_full[0, j] = pred_mean[0, 0]  # Keep the predicted milk value
            else:
                pred_full[0, j] = temp_row_scaled[j]
        
        
        curr_sequence = np.append(curr_sequence[1:], pred_full, axis=0)
        
        
        future_predictions.append(pred_mean[0, 0])
        prediction_std.append(pred_std[0, 0])
    
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    prediction_std = np.array(prediction_std).reshape(-1, 1)
    
    dummy_pred = np.zeros((len(future_predictions), n_features))
    dummy_pred[:, target_var_idx] = future_predictions.flatten()
    future_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, target_var_idx]
    
    dummy_std = np.zeros((len(prediction_std), n_features))
    dummy_std[:, target_var_idx] = prediction_std.flatten()
    future_std_rescaled = scaler.inverse_transform(dummy_std)[:, target_var_idx]
    
    return future_pred_rescaled, future_std_rescaled, future_months, future_years, future_seasons, future_precip, future_forecast_precip, future_gaps

def make_milk_production_forecast(model, last_sequence, scaler, n_future=16, 
                                 external_precip_forecasts=None, n_simulations=100):
    """
    Generate milk production forecasts using a trained model
    
    Parameters:
    -----------
    model : Keras model
        The trained LSTM model for prediction
    last_sequence : numpy array
        The last sequence of data used as the starting point
    scaler : MinMaxScaler
        The scaler used to normalize the data
    n_future : int, default=16
        Number of future time steps to predict
    external_precip_forecasts : DataFrame or dict, optional
        External precipitation forecasts to use. If DataFrame, should have columns 
        'year', 'month_num', and 'Forecasted Precipitation'. If dict, keys should be 
        (year, month) tuples and values should be forecasted precipitation values.
    n_simulations : int, default=100
        Number of Monte Carlo simulations for uncertainty estimation
        
    Returns:
    --------
    forecast_df : DataFrame
        Dataframe containing the forecast results
    """

    future_precip_forecast = {}


    last_date = pd.Timestamp(test_date_objects[-1])
    last_year = last_date.year
    last_month = last_date.month

    
    has_external_forecasts = False
    if external_precip_forecasts is not None:
        has_external_forecasts = True
        
        
        if isinstance(external_precip_forecasts, pd.DataFrame):
            
            precip_df = external_precip_forecasts
        elif isinstance(external_precip_forecasts, dict):
            
            future_precip_forecast = external_precip_forecasts.copy()
        else:
            
            print("Warning: external_precip_forecasts must be a DataFrame or dictionary. Ignoring.")
            has_external_forecasts = False

    
    for i in range(n_future):
        
        future_month = last_month + i + 1
        future_year = last_year
        while future_month > 12:
            future_month -= 12
            future_year += 1
            
        
        if (future_year, future_month) in future_precip_forecast:
            continue
        
        
        forecast_value = None
        
        if has_external_forecasts and isinstance(external_precip_forecasts, pd.DataFrame):
            
            matching_future_rows = external_precip_forecasts[
                (external_precip_forecasts["year"] == future_year) & 
                (external_precip_forecasts["month_num"] == future_month)
            ]
            
            if len(matching_future_rows) > 0 and not matching_future_rows["Forecasted Precipitation"].isna().all():
                
                forecast_value = matching_future_rows["Forecasted Precipitation"].values[0]
        
        
        if forecast_value is None:
            
            matching_rows = data[(data["year"] == future_year) & (data["month_num"] == future_month)]
            if len(matching_rows) > 0 and not matching_rows["Forecasted Precipitation"].isna().all():
                
                forecast_value = matching_rows["Forecasted Precipitation"].values[0]
            else:
                
                month_data = data[data['month_num'] == future_month]
                if len(month_data) > 0:
                    forecast_value = month_data['Forecasted Precipitation'].mean()
                else:
                    forecast_value = 0.0
        
        
        future_precip_forecast[(future_year, future_month)] = forecast_value

    
    future_pred, future_std, future_months, future_years, future_seasons, future_precip, future_forecast_precip, future_gaps = forecast_future_with_gap_tracking(
        model, last_sequence, n_future, scaler, target_var_idx, year_idx, month_idx, season_idx, 
        precip_idx, forecast_precip_idx, gap_idx, data_month_to_season, 
        future_precip_forecast=future_precip_forecast, n_simulations=n_simulations
    )
    
    
    future_dates = [f"{year}-{month:02d}" for year, month in zip(future_years, future_months)]
    future_date_objects = [pd.Timestamp(f"{year}-{month:02d}-01") for year, month in zip(future_years, future_months)]

    
    future_actuals = []
    for year, month in zip(future_years, future_months):
        
        matching_rows = data[(data["year"] == year) & (data["month_num"] == month)]
        if len(matching_rows) > 0:
            
            actual_value = matching_rows["amountmilked"].values[0]
            future_actuals.append(actual_value)
        else:
            
            future_actuals.append(None)

    
    forecast_df = pd.DataFrame({
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
        'Upper Bound (95%)': future_pred + 1.96 * future_std
    })
    
    forecast_df['Error'] = None
    forecast_df['Percent Error'] = None

    for i in range(len(forecast_df)):
        if forecast_df['Actual (if available)'].iloc[i] is not None:
            forecast_df['Error'].iloc[i] = forecast_df['Actual (if available)'].iloc[i] - forecast_df['Forecasted Amount Milked'].iloc[i]
            forecast_df['Percent Error'].iloc[i] = (forecast_df['Error'].iloc[i] / forecast_df['Actual (if available)'].iloc[i]) * 100
            
    return forecast_df

# 
n_future = 16
last_sequence = X_test[-1]  # Get the last sequence from test data


forecast_result_df = make_milk_production_forecast(model, last_sequence, scaler, 
                                                 external_precip_forecasts=precipitation_forecasts_df)

future_months = forecast_result_df['Month'].values
future_years = forecast_result_df['Year'].values 
future_seasons = forecast_result_df['Season_Index'].values
future_precip = forecast_result_df['Precipitation'].values
future_gaps = forecast_result_df['Months Gap'].values
future_pred = forecast_result_df['Forecasted Amount Milked'].values
future_std = forecast_result_df['Forecast Uncertainty (Std Dev)'].values
future_forecast_precip = None  


print("\nFuture Forecasts from function:")
print(forecast_result_df[['Date', 'Forecasted Amount Milked', 'Forecast Uncertainty (Std Dev)', 'Lower Bound (95%)', 'Upper Bound (95%)']])


future_dates = [f"{year}-{month:02d}" for year, month in zip(future_years, future_months)]
future_date_objects = [pd.Timestamp(f"{year}-{month:02d}-01") for year, month in zip(future_years, future_months)]


future_actuals = []
for year, month in zip(future_years, future_months):
    
    matching_rows = data[(data["year"] == year) & (data["month_num"] == month)]
    if len(matching_rows) > 0:
        
        actual_value = matching_rows["amountmilked"].values[0]
        future_actuals.append(actual_value)
    else:
        
        future_actuals.append(None)

forecast_df = pd.DataFrame({
    'Ward':unique_ward1,
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
    'Upper Bound (95%)': future_pred + 1.96 * future_std
})

forecast_df=forecast_df

forecast_df['Error'] = None
forecast_df['Percent Error'] = None

for i in range(len(forecast_df)):
    if forecast_df['Actual (if available)'].iloc[i] is not None:
        forecast_df['Error'].iloc[i] = forecast_df['Actual (if available)'].iloc[i] - forecast_df['Forecasted Amount Milked'].iloc[i]
        forecast_df['Percent Error'].iloc[i] = (forecast_df['Error'].iloc[i] / forecast_df['Actual (if available)'].iloc[i]) * 100

print("\nFuture Forecasts with Gap Tracking (original format):")
print(forecast_df)


plt.figure(figsize=(12, 6))


plt.plot(test_date_objects, y_test_rescaled, 'b-', label='Historical')


plt.plot(future_date_objects, future_pred, 'r-', label='Forecast')
plt.fill_between(
    future_date_objects,
    future_pred - 1.96 * future_std,
    future_pred + 1.96 * future_std,
    color='pink', alpha=0.3, label='95% Confidence Interval'
)


actuals_dates = []
actuals_values = []
for i, actual in enumerate(future_actuals):
    if actual is not None:
        actuals_dates.append(future_date_objects[i])
        actuals_values.append(actual)

if len(actuals_dates) > 0:
    plt.plot(actuals_dates, actuals_values, 'go', markersize=8, label='Actual Values')


plt.axvline(x=test_date_objects[-1], color='k', linestyle='--', label='Forecast Start')

plt.title('Milk Production Future Forecast with Uncertainty and Available Actuals')
plt.legend()
plt.ylabel('Amount Milked')
plt.xlabel('Date')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import matplotlib.dates as mdates


plt.figure(figsize=(12, 6))


actual_dates = []


latest_date = max(pd.to_datetime(future_date_objects[-1]), pd.to_datetime(test_date_objects[-1]))
cutoff_date = latest_date - pd.DateOffset(months=11)  # To get 12 months total (current + 11 previous)

all_actual_dates = [pd.to_datetime(date) for date in test_date_objects]
all_actual_values = y_test_rescaled.tolist()

for i, actual in enumerate(future_actuals):
    if actual is not None:
        all_actual_dates.append(pd.to_datetime(future_date_objects[i]))
        all_actual_values.append(actual)

actual_dates = []
actual_values = []
for i, date in enumerate(all_actual_dates):
    if date >= cutoff_date:
        actual_dates.append(date)
        actual_values.append(all_actual_values[i])


all_forecast_dates = [pd.to_datetime(date) for date in future_date_objects]
all_forecast_values = future_pred.tolist()
all_forecast_lower = [val - 1.96 * std for val, std in zip(future_pred, future_std)]
all_forecast_upper = [val + 1.96 * std for val, std in zip(future_pred, future_std)]


forecast_dates = []
forecast_values = []
forecast_lower = []
forecast_upper = []
for i, date in enumerate(all_forecast_dates):
    if date >= cutoff_date:
        forecast_dates.append(date)
        forecast_values.append(all_forecast_values[i])
        forecast_lower.append(all_forecast_lower[i])
        forecast_upper.append(all_forecast_upper[i])


display_forecast_dates = []
display_forecast_values = []
for i, date in enumerate(all_forecast_dates):
    if date >= cutoff_date and future_actuals[i] is None:
        display_forecast_dates.append(date)
        display_forecast_values.append(all_forecast_values[i])


if actual_dates:
    plt.plot(actual_dates, actual_values, 'b-o', markersize=5, label='Actual Values', zorder=3)


if forecast_dates:
    plt.fill_between(
        forecast_dates,
        forecast_lower,
        forecast_upper,
        color='pink', alpha=0.3, label='95% Confidence Interval',
        zorder=1
    )


if display_forecast_dates and actual_dates:

    last_actual_date = actual_dates[-1]
    last_actual_value = actual_values[-1]
    
   
    connected_dates = [last_actual_date] + display_forecast_dates
    connected_values = [last_actual_value] + display_forecast_values
    
  
    plt.plot(connected_dates, connected_values, 'o-', color='#FFA500', markersize=5, label='Forecast', zorder=2)


forecast_start_date = pd.to_datetime(future_date_objects[0])
if forecast_start_date >= cutoff_date:
    plt.axvline(x=forecast_start_date, color='k', linestyle='--', label='Forecast Start')


plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title('Milk Production: Last 12 Months with Future Forecast')


plt.legend(loc='lower left')

plt.ylabel('Amount Milked')
plt.xlabel('Month')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#forecast_df['Ward']=unique_ward1

forecast_df.rename(columns={'Months Gap': 'Months_Gap','Forecasted Amount Milked': 'Forecasted_Amount_Milked','Actual (if available)': 'Actual','Forecast Uncertainty (Std Dev)': 'Forecast_Uncertainty','Lower Bound (95%)': 'Lower_Bound','Upper Bound (95%)': 'Upper_Bound','Percent Error': 'Percent_Error'}, inplace=True)
forecast_df5=forecast_df[['Ward','Month','Year','Season_Index','Precipitation','Months_Gap','Date','Date_Object','Forecasted_Amount_Milked','Actual','Forecast_Uncertainty','Lower_Bound','Upper_Bound','Error','Percent_Error']]


import mysql.connector

conn = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='Romans17:48',
    database='livelihoodzones'
)

values = [
    tuple(None if (isinstance(x, float) and np.isnan(x)) else x for x in row)  
    for row in forecast_df5.itertuples(index=False, name=None)
]

insert_query = """
    INSERT INTO Predictions (
        Ward, Month, Year, Season_Index, Precipitation, 
        Months_Gap, Date, Date_Object, Forecasted_Amount_Milked, 
        Actual, Forecast_Uncertainty, Lower_Bound, Upper_Bound, 
        Error, Percent_Error
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

cursor = conn.cursor()
cursor.executemany(insert_query, values)
conn.commit()

cursor.close()
conn.close()


print(f"Finished processing \n")
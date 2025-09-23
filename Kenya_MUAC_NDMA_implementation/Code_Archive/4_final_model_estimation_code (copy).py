import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import os
from scipy.stats import norm
import re
import seaborn as sns

#------------------------------------------------------------------#
# Generating macros/lists with variable names for different models #
#------------------------------------------------------------------#
INPUT = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/intermediary_datasets"
SHAPE = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/shapefiles"
OUTPUT = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/results"
#---------------------------------------------------------------------------
# Change directory to the general folder that contains intermediary_datasets folder
os.chdir('/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation') 


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#------------------------------------------------------------------#
data = pd.read_pickle(os.path.join(INPUT, 
                                   'complete_ward_level_dataset_Jan_2021_to_Sep_2025_updated_on_Jun_2025.pkl'))

data['time_period'] = pd.to_datetime(
    data['year'].astype(str) + '-' + data['month'].astype(str).str.zfill(2),
    format='%Y-%m'
)
data['time_cont_enc'] = data['time_period'].rank(method='dense').astype(int)
#------------------------------------------------------------------#
# Generating macros/lists with variable names #
#------------------------------------------------------------------#
label_encoder_county = LabelEncoder()
label_encoder_livelihood = LabelEncoder()

# Apply label encoding to 'County' and 'LivelihoodZone'
data['County'] = label_encoder_county.fit_transform(data['County'])
data['LivelihoodZone'] = label_encoder_livelihood.fit_transform(data['LivelihoodZone'])


static = [ 'travel_time_to_cities_2015',
 'density_2015','density_2020','delta_2015_2020']

#===========================================
# Smoothing outcome var with gaussian kernel (trend capture)
#===========================================

def smooth_per_group(df, sigma, columns_to_smooth):
    smoothed_df = df.copy()
    window_size = int(3 * sigma) # Size of the Gaussian kernel (3 sigma covers over 99% of the curve)
    
    # Generate Gaussian kernel weights for the past values including current point (one-sided)
    # Note: We generate a kernel for each possible window size to ensure proper normalization at the start of the series
    kernels = {i: norm.pdf(np.arange(i, -1, -1), 0, sigma) for i in range(window_size + 1)}

    for unit_id in df['Ward'].unique():
        unit_data = df[df['Ward'] == unit_id].sort_values(by='time_cont_enc')       
        for col in columns_to_smooth:
            smoothed_col_name = f'{col}_smoothed'
            smoothed_values = np.zeros_like(unit_data[col].values)

            # One-sided Gaussian smoothing
            for i in range(len(unit_data)):
                # Determine the window size (less at the start of the series)
                current_window_size = min(i, window_size) + 1
                kernel = kernels[current_window_size - 1]
                kernel = kernel / kernel.sum()  # Normalize the kernel weights to sum to 1

                smoothed_values[i] = np.dot(unit_data[col].values[i - current_window_size + 1 : i + 1], kernel)
            
            smoothed_df.loc[unit_data.index, smoothed_col_name] = smoothed_values
    
    return smoothed_df


columns_to_smooth = ['wasting', 'wasting_risk']

data = smooth_per_group(data, sigma=3, columns_to_smooth=columns_to_smooth)


for lag in range(1, 13):  # Lags from 1 to 12 months
        # Generate the lagged data and store it in the dictionary
        data[f'wasting_sm_lag_{lag}'] = data.groupby('Ward')['wasting_smoothed'].shift(lag)
        data[f'wasting_sm_risk_lag_{lag}'] = data.groupby('Ward')['wasting_risk_smoothed'].shift(lag)

#==============================
#Getting dynamic variable lags
#==============================
globals_dict = globals()

# Iterate through specific lag numbers
for i in range(1, 13):  # Adjust the range as needed
    lag_pattern = f"lag_{i}"  # Create the exact lag pattern
    globals_dict[f"l{i}"] = [col for col in data if f"{lag_pattern}_" in col or col.endswith(
        lag_pattern)]
# generates lists of variables with the patterm l1, l2,....(on1-month lag, 2-month lag....)
#===========================================
# Dynamically create lists for each rain type
#===========================================
rain_types = ['long_rain', 'short_rain']

for rain_type in rain_types:
    globals()[rain_type] = [var for var in data if f"{rain_type}_max_prev" in var or f"{rain_type}_total_prev" in var or f"{rain_type}_avg_prev" in var]

for rain_type in rain_types:
    globals()[rain_type + '_lag'] = [
        var for var in data if f"{rain_type}_lag" in var]


#------------------------------------------------------------------#
# Generating datasets for each predictive horizon #
#------------------------------------------------------------------#
# Hybrid model #
# Wasting #

hb_m1 = (
    static +
    long_rain + short_rain +
    l1 + l2 + l3 )

hb_m2 = (
    static +
    long_rain + short_rain +
    l2 + l3 + l4 )

hb_m3 = (
    static +
    long_rain + short_rain +
    l3 + l4 + l5 )

#------------------------------


#--------------------------------------------------
num_cpus = os.cpu_count()

def bootstrap_iteration(model,train, testX, ff_m, num_iterations):
    all_predictions = np.zeros((num_iterations, testX.shape[0]))

    def single_bootstrap_iteration(iteration):
        resampled_data = train.groupby('time_cont_enc', group_keys=False).apply(
            lambda x: x.sample(frac=1, replace=True))

        X_sampled = resampled_data[[c for c in trainX.columns if c in ff_m]]
        y_sampled = resampled_data[outcome_var]

        model.fit(X_sampled, y_sampled)
        y_pred = model.predict(testX)

        return y_pred

    # Parallelize the bootstrap iterations
    all_predictions = Parallel(n_jobs=num_cpus)(
        delayed(single_bootstrap_iteration)(i) for i in tqdm(range(num_iterations))
    )

    return np.array(all_predictions)

#---------------------------------------------------------------------
def train_test_split(data_train, data_test, n_test, pred_h):
    i = n_test-36
    t = n_test + pred_h
    print(i, n_test, t)
    
    mask = (data_train['time_cont_enc']<n_test) & (data_train['time_cont_enc']>=i)
    
    return data[mask], data_test[data_test['time_cont_enc'] == t]
#---------------------------------------------------------------------

ph = [1, 2, 3]
model_name = ['hb']
outcome_var = "wasting_smoothed"
first_year_muac, first_month_muac = 2021, 1
last_year_muac, last_month_muac = 2025, 6

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

data_val = data.copy()

# Step 1: Find the last available time_cont_enc with outcome data
last_obs_month = data_val.loc[data_val[outcome_var].notna(), "time_cont_enc"].max()
feature_importances_file = os.path.join(results_dir, f"Feature_Importances_{outcome_var}_{last_obs_month}.csv")

# Create the CSV with headers if it doesn't exist
if not os.path.exists(feature_importances_file):
    pd.DataFrame(columns=['feature', 'importance', 
                          'model', 'time_horizon', 
                          'time_period']).to_csv(feature_importances_file, index=False)

for m in model_name:
    for horizon in ph:
        predictions_df = pd.DataFrame()
        globals()[f'feat_imp{horizon}'] = pd.DataFrame()
        globals()[f'feat_imp{horizon}']['var_names'] = globals()[f'{m}_m{horizon}']

        # Step 2: Training up to last_obs_month, and predicting for last_obs_month + horizon
        train, predict = train_test_split(data, data_val, last_obs_month, horizon)

        train_clean = train[train[outcome_var].notna()].copy() #in case there are missing values in the outcome variable in the last collected months
        trainX = train_clean[[c for c in train_clean.columns if c in globals()[f'{m}_m{horizon}']]]
        trainY = train_clean[outcome_var]
        testX = predict[[c for c in predict.columns if c in globals()[f'{m}_m{horizon}']]]

        model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1,
                             subsample=0.9, colsample_bylevel=1, seed=12345)
        model.fit(trainX, trainY)

        # Feature importance
        fi = pd.DataFrame({
            'feature': trainX.columns,
            'importance': model.feature_importances_,
            'model': m,
            'time_horizon': horizon,
            'time_period': predict["time_period"].iloc[0]
        })
        
        fi.to_csv(feature_importances_file, mode='a', header=False, index=False)

        # Prediction
        yhat = model.predict(testX)
        num_iterations = 1000
        all_predictions = bootstrap_iteration(model, train_clean, 
                                              testX, globals()[f'{m}_m{horizon}'], num_iterations)

        lower_bound = np.percentile(all_predictions, 5, axis=0)
        upper_bound = np.percentile(all_predictions, 95, axis=0)

        df_yhat = pd.DataFrame(data={'yhat': np.maximum(yhat, 0),
                                     'lower_bound': lower_bound,
                                     'upper_bound': upper_bound}, index=testX.index.copy())

        df_out = df_yhat.join(predict[["Ward", "time_cont_enc", "time_period"]])
        df_out['train_size'] = len(trainX)
        df_out['test_size'] = len(testX)

        predictions_df = pd.concat([predictions_df, df_out], ignore_index=True)

        # Save predictions for this horizon
        pred_file = f'results/{outcome_var}_pred_{m}_{horizon}_future_months.csv'
        predictions_df.to_csv(pred_file, index=False)
        globals()[f'predictions_{horizon}'] = predictions_df

# Merge with historic predictions and save new file
def merge_and_replace_predictions(results_dir="results", 
                                  model="hb", outcome_var="wasting_smoothed", horizons=[1, 2, 3]):
    for h in horizons:
        future_file = os.path.join(results_dir, f"{outcome_var}_pred_{model}_{h}_future_months.csv")

        # Match historical prediction file(s) like: outcome_pred_model_h_36m_YYYY_M_to_YYYY_M.csv
        pat_common = fr"{re.escape(outcome_var)}_pred_{re.escape(model)}_{h}_36m_\d+_\d+"
        pattern_with_to = re.compile(pat_common + r"_to_\d+_\d+\.csv$")
        pattern_no_to   = re.compile(pat_common + r"_\d+_\d+\.csv$")

        # List candidates and match either pattern
        all_files = os.listdir(results_dir)
        historic_files = [f for f in all_files if (
            pattern_with_to.fullmatch(f) or pattern_no_to.fullmatch(f))]

        if not historic_files:
            print(f"⚠️ No historical file found for horizon {h}. Skipping.")
            continue
        if not os.path.exists(future_file):
            print(f"⚠️ No future file found for horizon {h}. Skipping.")
            continue

        # If multiple historical files match, pick the most recently modified
        historic_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
        historic_file = os.path.join(results_dir, historic_files[0])

        print(f"🔄 Merging (keep future on overlap): {os.path.basename(historic_file)} + {os.path.basename(future_file)}")

        # Load
        df_hist = pd.read_csv(historic_file)
        df_future = pd.read_csv(future_file)

        # Ensure required columns exist and align
        # Add time_horizon if missing
        if "time_horizon" not in df_hist.columns:
            df_hist["time_horizon"] = h
        if "time_horizon" not in df_future.columns:
            df_future["time_horizon"] = h

        # Standardize time_period to datetime for keying (works even if already datetime-ish)
        df_hist["time_period"] = pd.to_datetime(df_hist["time_period"])
        df_future["time_period"] = pd.to_datetime(df_future["time_period"])

        key_cols = ["Ward", "time_period", "time_horizon"]

        # --- Keep future rows on overlap ---
        # Anti-join: drop from historical anything that appears in future keys
        future_keys = set(map(tuple, df_future[key_cols].to_records(index=False)))
        overlap_mask = df_hist[key_cols].apply(tuple, axis=1).isin(future_keys)
        n_overlap = int(overlap_mask.sum())
        if n_overlap > 0:
            print(f"🔁 Found {n_overlap} overlapping rows for horizon {h}. "
                  f"Keeping FUTURE rows and dropping those {n_overlap} from historical.")
        df_hist_no_overlap = df_hist.loc[~overlap_mask].copy()

        # Concat (now no duplicates by construction)
        df_merged = pd.concat([df_hist_no_overlap, df_future], ignore_index=True)

        # (Optional) sort for readability
        df_merged = df_merged.sort_values(["Ward", "time_period", "time_horizon"]).reset_index(drop=True)

        # New filename range
        min_date = df_merged["time_period"].min()
        max_date = df_merged["time_period"].max()
        new_filename = f"{outcome_var}_pred_{model}_{h}_36m_{min_date.year}_{min_date.month}_to_{max_date.year}_{max_date.month}.csv"
        new_filepath = os.path.join(results_dir, new_filename)

        # Save merged result
        df_merged.to_csv(new_filepath, index=False)
        print(f"✅ Saved merged file: {new_filename}")

        # Clean up old inputs
        os.remove(historic_file)
        print(f"🗑️  Deleted old file after merge: {os.path.basename(historic_file)}")
        os.remove(future_file)
        print(f"🗑️  Deleted future file after merge: {os.path.basename(future_file)}")

# Run it
merge_and_replace_predictions()


#=-----------------
# plotting top features
feature_importances_df = pd.read_csv(feature_importances_file)

horizon_df = feature_importances_df[feature_importances_df['time_horizon'] == 3]
last_3_months = sorted(horizon_df['time_period'].dropna().unique())[-3:]
# Build plot_df exactly as you already do
plot_data = []
for month in last_3_months:
    month_df = horizon_df[horizon_df['time_period'] == month]
    top10 = month_df.sort_values(by='importance', ascending=False).head(10).copy()
    top10['time_period'] = str(month)  # ensure string for labeling
    plot_data.append(top10)

plot_df = pd.concat(plot_data, ignore_index=True)

# One color per unique feature (works for any count of features)
unique_feats = plot_df['feature'].unique()
palette_list = sns.color_palette("husl", n_colors=len(unique_feats))
feature_colors = dict(zip(unique_feats, palette_list))

def facet_bar(data, **kws):
    # Order features by importance (ascending for nicer horizontal bars)
    order = data.sort_values('importance', ascending=True)['feature']
    ax = sns.barplot(
        data=data,
        x='importance',
        y='feature',
        order=order,
        hue='feature',           # color by feature
        dodge=False,             # don't split bars
        palette=feature_colors,  # consistent mapping
        orient='h',
        **kws
    )
    # Remove per-ax legend (feature names already on y-axis)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    return ax

g = sns.FacetGrid(plot_df, col="time_period", col_wrap=3, sharex=False, sharey=True, height=4, aspect=1.2)
g.map_dataframe(facet_bar)

# Cosmetics
g.set_titles("Top Features - {col_name}")
g.set_axis_labels("Importance", "Feature")

for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=3)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()

# Ensure output folder exists
os.makedirs("figures", exist_ok=True)

# Build a clean suffix from the months in the plot (e.g., 2025-06_2025-07_2025-08)
months = sorted(pd.to_datetime(plot_df['time_period'].unique()))
suffix = "_".join(m.strftime('%Y-%m') for m in months)

# Save (PNG for quick sharing, SVG for vector quality)
out_png = f"figures/top_features_{suffix}.png"
out_svg = f"figures/top_features_{suffix}.svg"

g.figure.savefig(out_png, dpi=300, bbox_inches='tight')
g.figure.savefig(out_svg, bbox_inches='tight')
print(f"Saved: {out_png}\nSaved: {out_svg}")

plt.show()




# If we want to plot other 3 specific months
target_months = ['2022-09-01', '2022-10-01', '2022-11-01']
plot_data = []
for month in target_months:
    month_df = horizon_df[horizon_df['time_period'] == month]
    top10 = month_df.sort_values(by='importance', ascending=False).head(10)
    top10 = top10.copy()
    top10['time_period'] = str(month)  # Ensure string for FacetGrid
    plot_data.append(top10)


plot_df = pd.concat(plot_data, ignore_index=True)

g = sns.FacetGrid(plot_df, col="time_period", col_wrap=3, sharex=False, sharey=True, height=4, aspect=1.2)
g.map_dataframe(sns.barplot, x="importance", y="feature", palette="Set2", orient='h')

g.set_titles("Top Features - {col_name}")
g.set_axis_labels("Importance", "Feature")

for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=3)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
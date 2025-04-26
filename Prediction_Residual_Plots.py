import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from scipy.stats import norm

# Create SQLAlchemy engine
engine = create_engine(
    'mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones'
)

def process_residual_plots(countyId, Indicator):

    query = """
        SELECT Predictions.Date_Object, Avg(Predictions.Last_Actual_Value) as Last_Actual_Value,Avg(Predictions.Month1_Forecast) as Month1_Forecast,Avg(Predictions.Month2_Forecast) as Month2_Forecast,Avg(Predictions.Month3_Forecast) as Month3_Forecast
        FROM (Predictions LEFT JOIN wards ON (Predictions.Ward = wards.Shapefile_wardName))LEFT JOIN subcounties ON (subcounties.SubCountyId = wards.SubCountyId) WHERE (subcounties.CountyId = %s AND Last_Actual_Value is not null AND Indicator = %s)
        GROUP BY Predictions.Date_Object
    """
    pooled_df = pd.read_sql(query, engine, params=(countyId, Indicator,))

    Forecast_df= pd.DataFrame({
        'Date_Object': pooled_df['Date_Object'],
        'Actual_Value': pooled_df['Last_Actual_Value'],
        # Align month1 forecast with actuals from next month
        '1_Month_Forecast': pooled_df['Month1_Forecast'].shift(+1),
        '1_Month_forecast_residuals': (pooled_df['Month1_Forecast'].shift(+1))-(pooled_df['Last_Actual_Value']),
        '2_Month_Forecast': pooled_df['Month2_Forecast'].shift(+2),
        '2_Month_forecast_residuals': (pooled_df['Month2_Forecast'].shift(+2))-(pooled_df['Last_Actual_Value']),    
        '3_Month_Forecast': pooled_df['Month3_Forecast'].shift(+3),
        '3_Month_forecast_residuals': (pooled_df['Month3_Forecast'].shift(+3))-(pooled_df['Last_Actual_Value'])    
    })
    #Forecast_df

    # ONE MONTH PREDICTION FORECAST RESIDUALS PLOT
    df = Forecast_df.copy()


    target_column = '1_Month_forecast_residuals'
    n_points = 200


    data = df[target_column].dropna().values
    if len(data) == 0:
        raise ValueError("No valid data points available after dropping NA values")


    mu, std = norm.fit(data)
    normal_dist = norm(loc=mu, scale=std)

    x_min, x_max = mu - 4*std, mu + 4*std  # Show ±4 standard deviations
    x_vals = np.linspace(x_min, x_max, n_points)

    plt.figure(figsize=(12, 6))

    # Plot normal distribution
    plt.plot(x_vals, normal_dist.pdf(x_vals), 
            color='#1f77b4', linewidth=3,
            label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')

    # Plot mean reference line
    plt.axvline(mu, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean = {mu:.2f}')

    # Plot mean reference line
    plt.axvline(0, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Zero = {0:.2f}')

    # Add standard deviation indicators
    plt.axvline(mu + std, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(mu - std, color='gray', linestyle=':', alpha=0.5)
    plt.text(mu + std, normal_dist.pdf(mu + std)*0.95, '+.5σ', ha='center')
    plt.text(mu - std, normal_dist.pdf(mu - std)*0.95, '-.5σ', ha='center')

    x_left_tail = np.linspace(x_min, mu - 0.95*std, 100)
    plt.fill_between(x_left_tail, normal_dist.pdf(x_left_tail), alpha=0.4, color='salmon')

    # Right tail (above +0.95σ)
    x_right_tail = np.linspace(mu + 0.95*std, x_max, 100)
    plt.fill_between(x_right_tail, normal_dist.pdf(x_right_tail), alpha=0.4, color='salmon')

    # Customize plot
    plt.title(f'Normality Density Plot for the 1-Month Milk Production Forecast Residuals {target_column}\n(n={len(data)})', pad=15)
    plt.xlabel('Residual Value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2, linestyle='--')

    # Add normality test results
    from scipy.stats import shapiro
    _, p_value = shapiro(data)
    plt.text(0.02, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}',
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('46_milkprod_1month_pred_Sep_2024.png')
    plt.show()



    # TWO MONTHS PREDICTION FORECAST RESIDUALS PLOT

    df = Forecast_df.copy()

    target_column = '2_Month_forecast_residuals'
    n_points = 200


    data = df[target_column].dropna().values
    if len(data) == 0:
        raise ValueError("No valid data points available after dropping NA values")


    mu, std = norm.fit(data)
    normal_dist = norm(loc=mu, scale=std)


    x_min, x_max = mu - 4*std, mu + 4*std  # Show ±4 standard deviations
    x_vals = np.linspace(x_min, x_max, n_points)


    plt.figure(figsize=(12, 6))

    # Plot normal distribution
    plt.plot(x_vals, normal_dist.pdf(x_vals), 
            color='#1f77b4', linewidth=3,
            label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')

    # Plot mean reference line
    plt.axvline(mu, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean = {mu:.2f}')

    # Plot mean reference line
    plt.axvline(0, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Zero = {0:.2f}')

    # Add standard deviation indicators
    plt.axvline(mu + std, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(mu - std, color='gray', linestyle=':', alpha=0.5)
    plt.text(mu + std, normal_dist.pdf(mu + std)*0.95, '+.5σ', ha='center')
    plt.text(mu - std, normal_dist.pdf(mu - std)*0.95, '-.5σ', ha='center')

    x_left_tail = np.linspace(x_min, mu - 0.95*std, 100)
    plt.fill_between(x_left_tail, normal_dist.pdf(x_left_tail), alpha=0.4, color='salmon')

    # Right tail (above +0.95σ)
    x_right_tail = np.linspace(mu + 0.95*std, x_max, 100)
    plt.fill_between(x_right_tail, normal_dist.pdf(x_right_tail), alpha=0.4, color='salmon')

    # Customize plot
    plt.title(f'Normality Density Plot for  Milk production {target_column}\n(n={len(data)})', pad=15)
    plt.xlabel('Residual Value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2, linestyle='--')

    # Add normality test results
    from scipy.stats import shapiro
    _, p_value = shapiro(data)
    plt.text(0.02, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}',
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('46_milkprod_2month_pred_Sep_2024.png')
    plt.show()



    # THREE MONTHS PREDICTION FORECAST RESIDUALS PLOT
    df = Forecast_df.copy()

    # Parameters
    target_column = '3_Month_forecast_residuals'
    n_points = 200

    # Clean data
    data = df[target_column].dropna().values
    if len(data) == 0:
        raise ValueError("No valid data points available after dropping NA values")

    # Calculate normal distribution parameters
    mu, std = norm.fit(data)
    normal_dist = norm(loc=mu, scale=std)

    # Create evaluation points
    x_min, x_max = mu - 4*std, mu + 4*std  # Show ±4 standard deviations
    x_vals = np.linspace(x_min, x_max, n_points)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot normal distribution
    plt.plot(x_vals, normal_dist.pdf(x_vals), 
            color='#1f77b4', linewidth=3,
            label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')

    # Plot mean reference line
    plt.axvline(mu, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean = {mu:.2f}')

    # Plot mean reference line
    plt.axvline(0, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Zero = {0:.2f}')

    # Add standard deviation indicators
    plt.axvline(mu + std, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(mu - std, color='gray', linestyle=':', alpha=0.5)
    plt.text(mu + std, normal_dist.pdf(mu + std)*0.95, '+.5σ', ha='center')
    plt.text(mu - std, normal_dist.pdf(mu - std)*0.95, '-.5σ', ha='center')

    x_left_tail = np.linspace(x_min, mu - 0.95*std, 100)
    plt.fill_between(x_left_tail, normal_dist.pdf(x_left_tail), alpha=0.4, color='salmon')

    # Right tail (above +0.95σ)
    x_right_tail = np.linspace(mu + 0.95*std, x_max, 100)
    plt.fill_between(x_right_tail, normal_dist.pdf(x_right_tail), alpha=0.4, color='salmon')

    # Customize plot
    plt.title(f'Normality Density Plot for  Milk production {target_column}\n(n={len(data)})', pad=15)
    plt.xlabel('Residual Value')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2, linestyle='--')

    # Add normality test results
    from scipy.stats import shapiro
    _, p_value = shapiro(data)
    plt.text(0.02, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}',
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('46_milkprod_3month_pred_Sep_2024.png')
    plt.show()


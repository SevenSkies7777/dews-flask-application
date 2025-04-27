import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from scipy.stats import norm
import requests
import os

# Create SQLAlchemy engine
engine = create_engine(
    'mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones'
)


# Upload plot image to Node.js backend
def upload_plot(filepath, filename):
  with open(filepath, 'rb') as f:
    files = {'file': (filename, f, 'image/png')}
    response = requests.post('http://127.0.0.1:2000/upload_images', files=files)

    if response.status_code == 200:
      print(f'✅ Uploaded {filename} successfully!')
      image_id = response.text.strip()
      print(f'🖼 Uploaded Image ID: {image_id}')
      return image_id  # <<< RETURN the image string
    else:
      print(f'❌ Failed to upload {filename}:', response.status_code,
            response.text)
      return None  # <<< Return None in case of failure


def process_residual_plots(countyId, Indicator):
  query = """
        SELECT Predictions.Date_Object, 
               AVG(Predictions.Last_Actual_Value) as Last_Actual_Value,
               AVG(Predictions.Month1_Forecast) as Month1_Forecast,
               AVG(Predictions.Month2_Forecast) as Month2_Forecast,
               AVG(Predictions.Month3_Forecast) as Month3_Forecast
        FROM (Predictions LEFT JOIN wards ON (Predictions.Ward = wards.Shapefile_wardName))
             LEFT JOIN subcounties ON (subcounties.SubCountyId = wards.SubCountyId)
        WHERE (subcounties.CountyId = %s AND Last_Actual_Value IS NOT NULL AND Indicator = %s)
        GROUP BY Predictions.Date_Object
    """
  pooled_df = pd.read_sql(query, engine, params=(countyId, Indicator,))

  Forecast_df = pd.DataFrame({
    'Date_Object': pooled_df['Date_Object'],
    'Actual_Value': pooled_df['Last_Actual_Value'],
    '1_Month_Forecast': pooled_df['Month1_Forecast'].shift(+1),
    '1_Month_forecast_residuals': pooled_df['Month1_Forecast'].shift(+1) -
                                  pooled_df['Last_Actual_Value'],
    '2_Month_Forecast': pooled_df['Month2_Forecast'].shift(+2),
    '2_Month_forecast_residuals': pooled_df['Month2_Forecast'].shift(+2) -
                                  pooled_df['Last_Actual_Value'],
    '3_Month_Forecast': pooled_df['Month3_Forecast'].shift(+3),
    '3_Month_forecast_residuals': pooled_df['Month3_Forecast'].shift(+3) -
                                  pooled_df['Last_Actual_Value']
  })

  # --- ONE MONTH FORECAST ---
  df = Forecast_df.copy()
  target_column = '1_Month_forecast_residuals'
  n_points = 200

  data = df[target_column].dropna().values
  if len(data) == 0:
    raise ValueError("No valid data points for 1 month")

  mu, std = norm.fit(data)
  normal_dist = norm(loc=mu, scale=std)
  x_min, x_max = mu - 4 * std, mu + 4 * std
  x_vals = np.linspace(x_min, x_max, n_points)

  plt.figure(figsize=(12, 6))
  plt.plot(x_vals, normal_dist.pdf(x_vals), color='#1f77b4', linewidth=3,
           label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')
  plt.axvline(mu, color='gray', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Mean = {mu:.2f}')
  plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Zero = {0:.2f}')
  plt.axvline(mu + std, color='gray', linestyle=':', alpha=0.5)
  plt.axvline(mu - std, color='gray', linestyle=':', alpha=0.5)
  plt.text(mu + std, normal_dist.pdf(mu + std) * 0.95, '+.5σ', ha='center')
  plt.text(mu - std, normal_dist.pdf(mu - std) * 0.95, '-.5σ', ha='center')
  x_left_tail = np.linspace(x_min, mu - 0.95 * std, 100)
  plt.fill_between(x_left_tail, normal_dist.pdf(x_left_tail), alpha=0.4,
                   color='salmon')
  x_right_tail = np.linspace(mu + 0.95 * std, x_max, 100)
  plt.fill_between(x_right_tail, normal_dist.pdf(x_right_tail), alpha=0.4,
                   color='salmon')
  plt.title(
      f'Normality Density Plot for the 1-Month Milk Production Forecast Residuals {target_column}\n(n={len(data)})',
      pad=15)
  plt.xlabel('Residual Value')
  plt.ylabel('Probability Density')
  plt.legend(loc='upper right')
  plt.grid(True, alpha=0.2, linestyle='--')
  from scipy.stats import shapiro
  _, p_value = shapiro(data)
  plt.text(0.02, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}',
           transform=plt.gca().transAxes,
           bbox=dict(facecolor='white', alpha=0.8))
  plt.tight_layout()

  one_month_file = '46_milkprod_1month_pred_Sep_2024.png'
  plt.savefig(one_month_file)
  one_month_img_id = upload_plot(one_month_file, one_month_file)
  os.remove(one_month_file)

  # --- TWO MONTH FORECAST ---
  df = Forecast_df.copy()
  target_column = '2_Month_forecast_residuals'
  data = df[target_column].dropna().values
  if len(data) == 0:
    raise ValueError("No valid data points for 2 month")

  mu, std = norm.fit(data)
  normal_dist = norm(loc=mu, scale=std)
  x_min, x_max = mu - 4 * std, mu + 4 * std
  x_vals = np.linspace(x_min, x_max, n_points)

  plt.figure(figsize=(12, 6))
  plt.plot(x_vals, normal_dist.pdf(x_vals), color='#1f77b4', linewidth=3,
           label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')
  plt.axvline(mu, color='gray', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Mean = {mu:.2f}')
  plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Zero = {0:.2f}')
  plt.axvline(mu + std, color='gray', linestyle=':', alpha=0.5)
  plt.axvline(mu - std, color='gray', linestyle=':', alpha=0.5)
  plt.text(mu + std, normal_dist.pdf(mu + std) * 0.95, '+.5σ', ha='center')
  plt.text(mu - std, normal_dist.pdf(mu - std) * 0.95, '-.5σ', ha='center')
  x_left_tail = np.linspace(x_min, mu - 0.95 * std, 100)
  plt.fill_between(x_left_tail, normal_dist.pdf(x_left_tail), alpha=0.4,
                   color='salmon')
  x_right_tail = np.linspace(mu + 0.95 * std, x_max, 100)
  plt.fill_between(x_right_tail, normal_dist.pdf(x_right_tail), alpha=0.4,
                   color='salmon')
  plt.title(
      f'Normality Density Plot for Milk Production 2-Month Forecast {target_column}\n(n={len(data)})',
      pad=15)
  plt.xlabel('Residual Value')
  plt.ylabel('Probability Density')
  plt.legend(loc='upper right')
  plt.grid(True, alpha=0.2, linestyle='--')
  _, p_value = shapiro(data)
  plt.text(0.02, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}',
           transform=plt.gca().transAxes,
           bbox=dict(facecolor='white', alpha=0.8))
  plt.tight_layout()

  two_month_file = '46_milkprod_2month_pred_Sep_2024.png'
  plt.savefig(two_month_file)
  two_month_img_id = upload_plot(two_month_file, two_month_file)
  os.remove(two_month_file)

  # --- THREE MONTH FORECAST ---
  df = Forecast_df.copy()
  target_column = '3_Month_forecast_residuals'
  data = df[target_column].dropna().values
  if len(data) == 0:
    raise ValueError("No valid data points for 3 month")

  mu, std = norm.fit(data)
  normal_dist = norm(loc=mu, scale=std)
  x_min, x_max = mu - 4 * std, mu + 4 * std
  x_vals = np.linspace(x_min, x_max, n_points)

  plt.figure(figsize=(12, 6))
  plt.plot(x_vals, normal_dist.pdf(x_vals), color='#1f77b4', linewidth=3,
           label=f'Normal Distribution (μ={mu:.2f}, σ={std:.2f})')
  plt.axvline(mu, color='gray', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Mean = {mu:.2f}')
  plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Zero = {0:.2f}')
  plt.axvline(mu + std, color='gray', linestyle=':', alpha=0.5)
  plt.axvline(mu - std, color='gray', linestyle=':', alpha=0.5)
  plt.text(mu + std, normal_dist.pdf(mu + std) * 0.95, '+.5σ', ha='center')
  plt.text(mu - std, normal_dist.pdf(mu - std) * 0.95, '-.5σ', ha='center')
  x_left_tail = np.linspace(x_min, mu - 0.95 * std, 100)
  plt.fill_between(x_left_tail, normal_dist.pdf(x_left_tail), alpha=0.4,
                   color='salmon')
  x_right_tail = np.linspace(mu + 0.95 * std, x_max, 100)
  plt.fill_between(x_right_tail, normal_dist.pdf(x_right_tail), alpha=0.4,
                   color='salmon')
  plt.title(
      f'Normality Density Plot for Milk Production 3-Month Forecast {target_column}\n(n={len(data)})',
      pad=15)
  plt.xlabel('Residual Value')
  plt.ylabel('Probability Density')
  plt.legend(loc='upper right')
  plt.grid(True, alpha=0.2, linestyle='--')
  _, p_value = shapiro(data)
  plt.text(0.02, 0.95, f'Shapiro-Wilk p-value: {p_value:.4f}',
           transform=plt.gca().transAxes,
           bbox=dict(facecolor='white', alpha=0.8))
  plt.tight_layout()

  three_month_file = '46_milkprod_3month_pred_Sep_2024.png'
  plt.savefig(three_month_file)
  three_month_img_id = upload_plot(three_month_file, three_month_file)
  os.remove(three_month_file)
  print(f'🖼 one_month_img_id: {one_month_img_id}')
  print(f'🖼 two_month_img_id: {two_month_img_id}')
  print(f'🖼 three_month_img_id: {three_month_img_id}')

  Session = sessionmaker(bind=engine)
  session = Session()

  try:
    # First, get the max date that meets your criteria
    max_date_query = text("""
            SELECT MAX(Predictions.Date_Object) 
            FROM Predictions 
            LEFT JOIN wards ON (Predictions.Ward = wards.Shapefile_wardName)
            LEFT JOIN subcounties ON (subcounties.SubCountyId = wards.SubCountyId) 
            WHERE subcounties.CountyId = :county_id AND Indicator = :indicator AND Predictions.Actual IS NOT NULL
        """)

    # max_date_result = session.execute(max_date_query).scalar()
    max_date_result = session.execute(max_date_query, {"county_id": countyId,
                                                       "indicator": Indicator}).scalar()

    if max_date_result:
      # Now perform the update on the Predictions table for rows with this date
      update_query = text("""
                UPDATE Predictions
                SET `1MonthResidID` = :new_value,`2MonthResidID` = :new_value2,`3MonthResidID` = :new_value3
                WHERE Date_Object = :max_date
                AND Ward IN (
                    SELECT wards.Shapefile_wardName
                    FROM wards
                    JOIN subcounties ON subcounties.SubCountyId = wards.SubCountyId
                    WHERE subcounties.CountyId = :county_id AND Indicator = :indicator
                )
                AND Actual IS NOT NULL
            """)

      # Execute the update
      result = session.execute(update_query,
                               {"new_value": one_month_img_id,
                                "new_value2": two_month_img_id,
                                "new_value3": three_month_img_id,
                                "county_id": countyId,
                                "indicator": Indicator,
                                "max_date": max_date_result}
                               )
      session.commit()

      print(f"Updated {result.rowcount} records")
    else:
      print("No matching records found")

  except Exception as e:
    session.rollback()
    print(f"An error occurred: {e}")

  finally:
    session.close()

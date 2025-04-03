import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import gc
from tqdm import tqdm
from sqlalchemy import create_engine
from pathlib import Path


class RainfallDataProcessor:
  """Class to process rainfall data from NetCDF and store it in MySQL."""

  def __init__(self, db_user, db_password, db_host, db_name):
    """Initialize database connection."""
    self.db_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
    self.engine = create_engine(self.db_url)

  def get_existing_dates(self):
    """Retrieve unique dates from MySQL Precipitation table."""
    query = "SELECT DISTINCT T FROM Precipitation"
    existing_dates = pd.read_sql(query, self.engine)
    if not existing_dates.empty:
      existing_dates["T"] = pd.to_datetime(existing_dates["T"]).dt.date
    return set(existing_dates["T"])

  def filter_new_precipitation_data(self, precipitation_df):
    """Filter out precipitation data that already exists in MySQL."""
    existing_dates = self.get_existing_dates()
    precipitation_df["T"] = pd.to_datetime(precipitation_df["T"]).dt.date
    new_data = precipitation_df[
      ~precipitation_df["T"].isin(existing_dates)].copy()
    print(f"Original data: {len(precipitation_df)} rows")
    print(f"New data: {len(new_data)} rows")
    return new_data

  def save_to_mysql(self, batch_df):
    """Save batch data to MySQL."""
    batch_df.to_sql(name="Precipitation", con=self.engine, if_exists="append",
                    index=False)
    print(f"Saved {len(batch_df)} records to MySQL.")

  def process_ward(self, precipitation_df, wards_gdf, ward_name):
    """Process rainfall data for a single ward."""
    try:
      ward_data = wards_gdf[wards_gdf['NAME_3'] == ward_name].copy()

      geometry = [Point(xy) for xy in
                  zip(precipitation_df.X, precipitation_df.Y)]
      precip_gdf = gpd.GeoDataFrame(precipitation_df, geometry=geometry,
                                    crs="EPSG:4326")

      ward_data = ward_data.to_crs("EPSG:4326")

      joined_data = gpd.sjoin(precip_gdf, ward_data, how="inner",
                              predicate="within")
      joined_data['T'] = pd.to_datetime(joined_data['T']).dt.date

      selected_columns = ['Y', 'X', 'T', 'precipitation', 'COUNTRY', 'NAME_1',
                          'NAME_2', 'NAME_3']
      joined_data = joined_data[selected_columns]

      return joined_data

    except Exception as e:
      print(f"Error processing ward {ward_name}: {str(e)}")
      return None

  def process_wards_batch(self, precipitation_df, wards_gdf, ward_names,
      batch_size=5):
    """Process wards in small batches and save results into MySQL."""
    batch_count = 0

    for i in tqdm(range(0, len(ward_names), batch_size)):
      batch_wards = ward_names[i:i + batch_size]
      batch_results = []

      for ward_name in batch_wards:
        try:
          result = self.process_ward(precipitation_df, wards_gdf, ward_name)
          if result is not None and not result.empty:
            batch_results.append(result)
            print(f"Successfully processed ward: {ward_name}")
        except Exception as e:
          print(f"Failed to process ward {ward_name}: {str(e)}")
          continue

      if batch_results:
        batch_df = pd.concat(batch_results, ignore_index=True)
        self.save_to_mysql(batch_df)
        print(f"Saved batch {batch_count} to database")

        del batch_results, batch_df
        gc.collect()

      batch_count += 1
      gc.collect()

  def processRainfallData(self, nc_file_path, shapefile_path,
      specific_wards=None):
    """Public method to process rainfall data (entry point)."""
    try:
      # Load shapefile
      wards_gdf = gpd.read_file(shapefile_path)

      # Load NetCDF data
      data = xr.open_dataset(nc_file_path, decode_times=False)
      ref_date = pd.Timestamp('1960-01-01')
      time_in_months = data['T'].values
      dates = [ref_date + pd.DateOffset(months=int(month)) for month in
               time_in_months]
      data = data.assign_coords(T=("T", dates))
      precipitation_df = data.to_dataframe().reset_index()

      # Filter new precipitation data
      precipitation_df = self.filter_new_precipitation_data(precipitation_df)

      # Get ward names
      if specific_wards is None:
        ward_names = wards_gdf['NAME_3'].unique().tolist()
      else:
        ward_names = specific_wards

      # Process and save data
      if not precipitation_df.empty:
        self.process_wards_batch(
            precipitation_df=precipitation_df,
            wards_gdf=wards_gdf,
            ward_names=ward_names,
            batch_size=5
        )

    except Exception as e:
      print(f"Error in processRainfallData: {str(e)}")

  def close(self):
    """Close the SQLAlchemy engine."""
    self.engine.dispose()
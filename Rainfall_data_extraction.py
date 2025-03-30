#importing libraries
import pandas as pd
import numpy as np
import mysql.connector
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from datetime import timedelta

conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Romans17:48',
        database='livelihoodzones'
    )

cursor = conn.cursor()

query = """
    SELECT wards.Shapefile_wardName
    FROM wards
   WHERE (Shapefile_wardName is not null)
    """

df = pd.read_sql(query, conn)
conn.close()
#df

#Extracting Rainfall Data from NC FIle
import xarray as xr
import numpy as np
import pandas as pd

data = xr.open_dataset(r'/home/ebenezer/Desktop/NDMADEWS_ML_DS/Pr.nc', decode_times=False)

ref_date = pd.Timestamp('1960-01-01')

time_in_months = data['T'].values
dates = [ref_date + pd.DateOffset(months=int(month)) for month in time_in_months]
data = data.assign_coords(T=("T", dates))

precipitation_df1 = data.to_dataframe().reset_index()

print(precipitation_df1.head())
column_counts = precipitation_df1.count()
print("Count of entries per column:\n", column_counts)

wards_df = df['Shapefile_wardName'].unique().tolist()
wards_df


conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Romans17:48',
        database='livelihoodzones'
    )

cursor = conn.cursor()

#Check if the new loaded NC file has new rainfall data

def get_existing_dates(conn):
    """Retrieve unique dates from the MySQL Precipitation table."""
    query = "SELECT DISTINCT T FROM Precipitation"
    existing_dates = pd.read_sql(query, conn) 
    if not existing_dates.empty:
        existing_dates["T"] = pd.to_datetime(existing_dates["T"]).dt.date  
    return set(existing_dates["T"])  

def filter_new_precipitation_data(precipitation_df, conn):
    """Create precipitation_df1 with only new dates not present in the MySQL database."""
    existing_dates = get_existing_dates(conn)  # Fetch saved dates
    precipitation_df1["T"] = pd.to_datetime(precipitation_df1["T"]).dt.date  # Ensure date format

    # Keep only rows where date is not in existing_dates
    precipitation_df = precipitation_df1[~precipitation_df1["T"].isin(existing_dates)].copy()



    print(f"Original data: {len(precipitation_df1)} rows")
    print(f"New data (precipitation_df1): {len(precipitation_df)} rows")
    return precipitation_df

precipitation_df = filter_new_precipitation_data(precipitation_df1, conn)


#Spatial Analysis of rainfall data against Shapefile
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import gc  
import mysql.connector

def save_to_mysql(batch_df, conn):
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO Precipitation (Y, X, T, precipitation, COUNTRY, NAME_1, NAME_2, NAME_3)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = batch_df.to_records(index=False).tolist()
    cursor.executemany(insert_query, values)
    conn.commit()
    cursor.close()

def process_ward(precipitation_df, wards_gdf, ward_name):
    """
    Process a single ward using the original working approach.
    """
    try:

        ward_data = wards_gdf[wards_gdf['NAME_3'] == ward_name].copy()

        geometry = [Point(xy) for xy in zip(precipitation_df.X, precipitation_df.Y)]
        precip_gdf = gpd.GeoDataFrame(
            precipitation_df, 
            geometry=geometry, 
            crs="EPSG:4326"
        )

        ward_data = ward_data.to_crs("EPSG:4326")

        joined_data = gpd.sjoin(precip_gdf, ward_data, how="inner", predicate="within")

        joined_data['T'] = pd.to_datetime(joined_data['T']).dt.date  # Ensure YYYY-MM-DD format

        selected_columns = ['Y', 'X', 'T', 'precipitation', 'COUNTRY', 'NAME_1', 'NAME_2', 'NAME_3']
        joined_data = joined_data[selected_columns]

        print(f"Joined data for ward {ward_name}:")
        print(joined_data.head())
        
        # Clean up
        del precip_gdf
        del ward_data
        gc.collect()
        
        return joined_data
        
    except Exception as e:
        print(f"Error processing ward {ward_name}: {str(e)}")
        return None

def process_wards_batch(precipitation_df, wards_gdf, ward_names, conn, batch_size=5):
    """
    Process wards in small batches and save results into MySQL database.
    """
    batch_count = 0
    
    # Process wards in batches
    for i in tqdm(range(0, len(ward_names), batch_size)):
        batch_wards = ward_names[i:i + batch_size]
        batch_results = []
        
        for ward_name in batch_wards:
            try:
                result = process_ward(precipitation_df, wards_gdf, ward_name)
                if result is not None and not result.empty:
                    batch_results.append(result)
                    print(f"Successfully processed ward: {ward_name}")
            except Exception as e:
                print(f"Failed to process ward {ward_name}: {str(e)}")
                continue
        
        # If we have results in this batch, the results are saved to MySQL database using direct connection
        if batch_results:
            batch_df = pd.concat(batch_results, ignore_index=True)
            

            save_to_mysql(batch_df, conn)
            print(f"Saved batch {batch_count} to database")
            
            # Clean up
            del batch_results
            del batch_df
            gc.collect()
            
            batch_count += 1
        
        # garbage collection between batches
        gc.collect()

# Call usage
def main(nc_file_path, shapefile_path, conn, specific_wards=None):
    """
    Main function to run the analysis.
    """
    try:

        wards_gdf = gpd.read_file(shapefile_path)
        

        if specific_wards is None:
            ward_names = wards_gdf['NAME_3'].unique().tolist()
        else:
            ward_names = specific_wards
        
        

        if not precipitation_df1.empty:
            process_wards_batch(
            precipitation_df=precipitation_df,  
            wards_gdf=wards_gdf,
            ward_names=ward_names,
            conn=conn,
            batch_size=5  # Batch Size for processing and loading. Can be adjusted based on system's memory
        )
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        

if __name__ == "__main__":
   
    nc_file_path = '/home/ebenezer/Desktop/NDMADEWS_ML_DS/Pr.nc'
    shapefile_path = '/home/ebenezer/Desktop/NDMADEWS_ML_DS/GADM/gadm41_KEN_3.shp'
    

conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Romans17:48',
        database='livelihoodzones'
    )

cursor = conn.cursor()
    

specific_wards = wards_df
main(nc_file_path, shapefile_path, conn, specific_wards)
    

conn.close()

# Created on April of 2025
# By: Susana Constenla
# Code to generate the secondary variable dataset (predictors)

#------------------------------------------
#==============================================================
import pandas as pd
import os, geemap, ee, glob, time, re, shutil
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
#-----------------
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
#------------------
import numpy as np
import argparse

#==============================================================
# Set input and output data path
SHAPE = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/shapefiles"
OUTPUT = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/intermediary_datasets"
#==============================================================
 # Getting ward polygons in our dataset

#muac = gpd.GeoDataFrame(pd.read_pickle(
#    os.path.join(OUTPUT,
#    'Kenya_NDMA_MUAC_23_counties_Mar_2025_WARD_LEVEL.pkl')))  #read dataset generated in last code snippet /// 
                                                                       #(ward level dataset with last collected data)
# if __name__ == "__main__":
#     args = parser.parse_args()
    
# Load polygons from the provided pickle file
polygons = gpd.read_file(
    os.path.join(SHAPE, 
    'Kenya_wards_NDMA.shp'))

#==========================
class SecondaryVariableGenerator:
    def __init__(self, shape_path, output_path, polygon_id_col, 
                 start_date_main, end_date_main,
                 start_date_evi, end_date_evi, polygons,
                 years_pop_den):
        
        self.SHAPE = shape_path
        self.OUTPUT = output_path
        self.polygon_id_col = polygon_id_col
        self.polygons = polygons

        # Date ranges
        self.start_date_main = start_date_main
        self.end_date_main = end_date_main
        self.start_date_evi = start_date_evi
        self.end_date_evi = end_date_evi
        self.years_pop_den = years_pop_den

        # Initialize EE
        # ee.Authenticate()
        ee.Initialize()

    #----------------- UPLOAD HELPER FUNCTIONS -----------------#

    def extract_years_months(self, start_date, end_date):
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        months = pd.date_range(start=start, end=end, freq='MS').strftime("%Y-%m").tolist()
        return [(int(month.split('-')[0]), int(month.split('-')[1])) for month in months]

    def consecutive_days(self, image, previous):
        '''Function to get consecutive day stats of different variables '''
        previous = ee.Image(previous)
        return image.add(previous.multiply(image))

    def wait_for_gee_tasks_to_finish(self, poll_interval=30):
        print("* Waiting for Earth Engine tasks to complete...")

        while True:
            try:
                tasks = ee.batch.Task.list()
            except Exception as e:
                print(f"[!] Error while listing tasks: {e}. Retrying in {poll_interval} seconds...")
                time.sleep(poll_interval)
                continue

            running_or_ready = [t for t in tasks if t.state in ['RUNNING', 'READY']]
            print(f"[⏳] {len(running_or_ready)} tasks still running...")

            if not running_or_ready:
                print("[✅] All tasks finished.")
                break

            time.sleep(poll_interval)


    #----------------- PIPELINES -----------------#
#============================================================
#--------------- CHIRPS PRECIPITATION (5KM) --------------------#
#============================================================
# --------- 1. Get CHIRPS monthly stats ---------
    def get_chirps_precipitation_by_month(self, year, month, ee_polygons):
            chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filter(ee.Filter.calendarRange(year, year, 'year')) \
                .filter(ee.Filter.calendarRange(month, month, 'month'))

            total_precip = chirps.reduce(ee.Reducer.sum()).rename(['precip_total'])
            wet_days = chirps.map(lambda img: img.gte(1)).reduce(ee.Reducer.sum()).rename(['wet_days'])
            dry_days = chirps.map(lambda img: img.lt(1)).reduce(ee.Reducer.sum()).rename(['dry_days'])

            def consecutive_days(img, prev):
                prev = ee.Image(prev)
                return img.multiply(prev.add(1)).rename('consec')

            wet_binary = chirps.map(lambda img: img.gte(1))
            max_consec_wet = ee.Image(wet_binary.iterate(consecutive_days, ee.Image(0))).reduce(ee.Reducer.max()).rename('consec_wet_days')

            dry_binary = chirps.map(lambda img: img.lt(1))
            max_consec_dry = ee.Image(dry_binary.iterate(consecutive_days, ee.Image(0))).reduce(ee.Reducer.max()).rename('consec_dry_days')

            combined = total_precip.addBands(wet_days).addBands(dry_days).addBands(max_consec_wet).addBands(max_consec_dry)

            stats = combined.reduceRegions(
                collection=ee_polygons,
                reducer=ee.Reducer.mean(),
                scale=5000
            ).map(lambda f: f.set('year', year).set('month', month))

            return stats

    def export_chirps_in_chunks(self,
            year, month,
            polygons_gdf,
            polygon_id_col,
            chunk_size=10,
            scale=5000,
            folder_name="chirps_exports",
            description_prefix="chirps_chunk"
        ):
            print(f"📅 Exporting CHIRPS for {year}-{month:02d} in chunks of {chunk_size} polygons")

            polygons_gdf = polygons_gdf[~polygons_gdf.geometry.is_empty]
            polygons_gdf = polygons_gdf[polygons_gdf.is_valid]
            polygons_gdf = polygons_gdf.to_crs("EPSG:4326")

            def chunk_polygons(gdf, chunk_size=10):
                for i in range(0, len(gdf), chunk_size):
                    yield i, gdf.iloc[i:i + chunk_size]

            for idx, chunk in chunk_polygons(polygons_gdf, chunk_size):
                ee_polygons = geemap.geopandas_to_ee(chunk[[self.polygon_id_col, 'geometry']])
                try:
                    stats = self.get_chirps_precipitation_by_month(year, month, ee_polygons)
                    size = stats.size().getInfo()

                    if size == 0:
                        print(f"[✗] Chunk {idx} has no data, skipping.")
                        continue

                    task = ee.batch.Export.table.toDrive(
                        collection=stats,
                        description=f"{description_prefix}_{year}_{month:02d}_chunk{idx}",
                        folder=folder_name,
                        fileNamePrefix=f"{description_prefix}_{year}_{month:02d}_chunk{idx}",
                        fileFormat='CSV'
                    )
                    task.start()
                    print(f"[✓] Task submitted: {year}-{month:02d} | chunk {idx} | features: {size}")

                except Exception as e:
                    print(f"[✗] Failed to export chunk {idx}: {e}")

        # -------------------

    
    def download_from_drive_chirps(self, output_folder='chirps_exports',
                                    drive_folder_name='chirps_exports'):
                os.makedirs(output_folder, exist_ok=True)
                gauth = GoogleAuth()
                gauth.LocalWebserverAuth()
                drive = GoogleDrive(gauth)

                # Find all folders with the given name
                folder_list = drive.ListFile({
                    'q': f"title = '{drive_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
                }).GetList()

                if not folder_list:
                    raise Exception(f"[✗] Folder '{drive_folder_name}' not found in your Google Drive.")

                print(f"[✓] Found {len(folder_list)} folders named '{drive_folder_name}'.")

                all_files = []
                for folder in folder_list:
                    folder_id = folder['id']
                    query = f"'{folder_id}' in parents and trashed=false and title contains 'chirps_chunk' and mimeType != 'application/vnd.google-apps.folder'"
                    files = drive.ListFile({'q': query}).GetList()
                    all_files.extend(files)

                print(f"[✓] Found {len(all_files)} total files across all '{drive_folder_name}' folders.")

                for file in tqdm(all_files, desc="Downloading from Drive"):
                    filename = file['title']
                    file_path = os.path.join(output_folder, filename)

                    try:
                        file.GetContentFile(file_path)
                    except Exception as e:
                        print(f"[✗] Failed to download {filename}: {e}")

                for file in tqdm(all_files, desc="Deleting from Drive"):
                    try:
                        file.Delete()
                    except Exception as e:
                        print(f"[!] Could not delete {file['title']}: {e}")


                for folder in folder_list:
                    try:
                        folder.Delete()
                        print(f"[✓] Deleted folder: {folder['title']} ({folder['id']})")
                    except Exception as e:
                        print(f"[!] Could not delete folder {folder['title']}: {e}")


            # -------------------
    
    def combine_csv_exports_and_save_chirps(self, csv_dir, output_pkl):
                all_csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
                print(f"Found {len(all_csvs)} CSV files...")

                df_list = []
                for file in tqdm(all_csvs, desc="Reading CSVs"):
                    try:
                        df = pd.read_csv(file)
                        df_list.append(df)
                        os.remove(file)
                    except Exception as e:
                        print(f"Failed to read {file}: {e}")

                if df_list:
                    final_df = pd.concat(df_list, ignore_index=True)
                    final_df.sort_values(by=['year', 'month'], inplace=True)
                    final_df.to_pickle(output_pkl)
                    print(f"[✓] Data saved to {output_pkl}")
                    return final_df
                else:
                    print("[✗] No CSVs to combine.")
                    return pd.DataFrame()

            # -------------------
        
    def run_full_chirps_pipeline(self,start_date, end_date, polygons_gdf, polygon_id_col, output_pkl):
                print("[1] Exporting CHIRPS precipitation stats to Drive...")

                year_months = self.extract_years_months(start_date, end_date)
                for year, month in year_months:
                    self.export_chirps_in_chunks(year, month, polygons_gdf, polygon_id_col=polygon_id_col)

                self.wait_for_gee_tasks_to_finish()

                print("[2] Downloading CSVs from Drive...")
                self.download_from_drive_chirps()

                print("[3] Combining and saving final dataset...")
                df = self.combine_csv_exports_and_save_chirps("chirps_exports", output_pkl)

                shutil.rmtree('chirps_exports', ignore_errors=True)
                print("[✓] CHIRPS pipeline complete.")
                return df

    def run_chirps_pipeline(self):
            output_pkl = f"{self.OUTPUT}/chirps_stats_{self._format_dates(self.start_date_main, self.end_date_main)}.pkl"
            return self.run_full_chirps_pipeline(
                self.start_date_main, self.end_date_main, self.polygons, self.polygon_id_col, output_pkl)

#============================================================
#--------------- ERA5 TEMPERATURE (10KM) --------------------#
#============================================================
        
    def get_era5_temperature_by_month(self,year, month, polygons_ee):
        era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .filter(ee.Filter.calendarRange(month, month, 'month'))

        max_temp = era5.select('temperature_2m_max').map(lambda img: img.subtract(273.15).rename('max_temp'))
        min_temp = era5.select('temperature_2m_min').map(lambda img: img.subtract(273.15).rename('min_temp'))
        avg_temp = era5.select('temperature_2m').map(lambda img: img.subtract(273.15).rename('avg_temp'))

        avg_temp_month = avg_temp.mean().rename('avg_temp_month')
        hot_days = max_temp.map(lambda img: img.gt(30)).reduce(ee.Reducer.sum()).rename('hot_days')
        cold_days = min_temp.map(lambda img: img.lt(10)).reduce(ee.Reducer.sum()).rename('cold_days')

        consecutive_hot = max_temp.map(lambda img: img.gt(30))
        max_consec_hot = ee.Image(consecutive_hot.iterate(
             self.consecutive_days, ee.Image(0))).reduce(ee.Reducer.max()).rename('consec_hot_days')

        consecutive_cold = min_temp.map(lambda img: img.lt(10))
        max_consec_cold = ee.Image(consecutive_cold.iterate(
             self.consecutive_days, ee.Image(0))).reduce(ee.Reducer.max()).rename('consec_cold_days')

        combined = avg_temp_month.addBands(hot_days).addBands(cold_days).addBands(max_consec_hot).addBands(max_consec_cold)

        stats = combined.reduceRegions(
            collection=polygons_ee,
            reducer=ee.Reducer.mean(),
            scale=11132
        ).map(lambda f: f.set('year', year).set('month', month))

        return stats

    def export_era5_in_chunks(self, year, month, polygons_gdf, polygon_id_col="Ward",
                            chunk_size=10, scale=11132, folder_name="era5_exports"):
        print(f"Exporting ERA5 for {year}-{month:02d} in chunks of {chunk_size} polygons")

        polygons_gdf = polygons_gdf[~polygons_gdf.geometry.is_empty]
        polygons_gdf = polygons_gdf[polygons_gdf.is_valid]
        polygons_gdf = polygons_gdf.to_crs("EPSG:4326")

        def chunk_polygons(gdf, chunk_size=10):
            for i in range(0, len(gdf), chunk_size):
                yield i, gdf.iloc[i:i + chunk_size]

        for idx, chunk in chunk_polygons(polygons_gdf, chunk_size):
            ee_polygons = geemap.geopandas_to_ee(chunk[[self.polygon_id_col, 'geometry']])
            try:
                stats = self.get_era5_temperature_by_month(year, month, ee_polygons)
                size = stats.size().getInfo()

                if size == 0:
                    print(f"[✗] Chunk {idx} has no data, skipping.")
                    continue

                task = ee.batch.Export.table.toDrive(
                    collection=stats,
                    description=f"era5_y{year}_m{month:02d}_chunk{idx}",
                    folder=folder_name,
                    fileNamePrefix=f"era5_y{year}_m{month:02d}_chunk{idx}",
                    fileFormat='CSV'
                )
                task.start()
                print(f"[✓] Task submitted: {year}-{month:02d} | chunk {idx} | features: {size}")

            except Exception as e:
                print(f"[✗] Failed to export chunk {idx}: {e}")

    def download_from_drive_era5(self, output_folder='era5_exports', drive_folder_name='era5_exports'):
        os.makedirs(output_folder, exist_ok=True)
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        # Find all folders with the given name
        folder_list = drive.ListFile({
            'q': f"title = '{drive_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        }).GetList()

        if not folder_list:
            raise Exception(f"[✗] Folder '{drive_folder_name}' not found in your Google Drive.")

        print(f"[✓] Found {len(folder_list)} folders named '{drive_folder_name}'.")

        all_files = []
        for folder in folder_list:
            folder_id = folder['id']
            query = f"'{folder_id}' in parents and trashed=false and title contains 'era5_y' and mimeType != 'application/vnd.google-apps.folder'"
            files = drive.ListFile({'q': query}).GetList()
            all_files.extend(files)

        print(f"[✓] Found {len(all_files)} total files across all '{drive_folder_name}' folders.")

        for file in tqdm(all_files, desc="Downloading from Drive"):
            filename = file['title']
            file_path = os.path.join(output_folder, filename)
            try:
                file.GetContentFile(file_path)
            except Exception as e:
                print(f"[✗] Failed to download {filename}: {e}")

        for file in tqdm(all_files, desc="Deleting from Drive"):
            try:
                file.Delete()
            except Exception as e:
                print(f"[!] Could not delete {file['title']}: {e}")

        # Delete all folders after their files are deleted
        for folder in folder_list:
            try:
                folder.Delete()
                print(f"[✓] Deleted folder: {folder['title']} ({folder['id']})")
            except Exception as e:
                print(f"[!] Could not delete folder {folder['title']}: {e}")

    def combine_csv_exports_and_save_era5(self, csv_dir, output_pkl):
        all_csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
        print(f"Found {len(all_csvs)} CSV files...")

        df_list = []
        for file in tqdm(all_csvs, desc="Reading CSVs"):
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                os.remove(file)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_df.sort_values(by=['year', 'month'], inplace=True)
            final_df.to_pickle(output_pkl)
            print(f"[✓] Data saved to {output_pkl}")
            return final_df
        else:
            print("[✗] No CSVs to combine.")
            return pd.DataFrame()

    def run_full_era5_pipeline(self,start_date, end_date, polygons_gdf, polygon_id_col, output_pkl):
        print("[1] Exporting ERA5 temperature stats to Drive...")

        year_months = self.extract_years_months(start_date, end_date)
        for year, month in year_months:
            self.export_era5_in_chunks(year, month, polygons_gdf, polygon_id_col)

        self.wait_for_gee_tasks_to_finish()

        print("[2] Downloading CSVs from Drive...")
        self.download_from_drive_era5()

        print("[3] Combining and saving final dataset...")
        df = self.combine_csv_exports_and_save_era5("era5_exports", output_pkl)

        shutil.rmtree('era5_exports', ignore_errors=True)
        print("[✓] ERA5 pipeline complete.")
        return df

    def run_era5_pipeline(self):
            output_pkl = f"{self.OUTPUT}/era5_stats_{self._format_dates(self.start_date_main, self.end_date_main)}.pkl"
            return self.run_full_era5_pipeline(self.start_date_main, 
                                               self.end_date_main, 
                                               self.polygons, self.polygon_id_col, output_pkl)

#============================================================
#--------------- EVI/NDVI (250m) --------------------#
#============================================================
    def get_modis_ndvi_evi_by_month(self, year, month, polygons_ee):
        modis = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .filter(ee.Filter.calendarRange(month, month, 'month')) \
            .select(['NDVI', 'EVI'])

        monthly_avg = modis.mean().rename(['NDVI_mean', 'EVI_mean'])

        stats = monthly_avg.reduceRegions(
            collection=polygons_ee,
            reducer=ee.Reducer.mean(),
            scale=250
        ).map(lambda f: f.set('year', year).set('month', month))

        return stats

# 2. Export to Drive in chunks
    def export_modis_in_chunks(self,
        year, month,
        polygons_gdf,
        polygon_id_col="Ward",
        chunk_size=10,
        scale=250,
        folder_name="modis_exports"
    ):
        print(f"📅 Exporting MODIS for {year}-{month:02d} in chunks of {chunk_size} polygons")

        polygons_gdf = polygons_gdf[~polygons_gdf.geometry.is_empty]
        polygons_gdf = polygons_gdf[polygons_gdf.is_valid]
        polygons_gdf = polygons_gdf.to_crs("EPSG:4326")

        def chunk_polygons(gdf, chunk_size):
            for i in range(0, len(gdf), chunk_size):
                yield i, gdf.iloc[i:i + chunk_size]

        for idx, chunk in chunk_polygons(polygons_gdf, chunk_size):
            ee_polygons = geemap.geopandas_to_ee(chunk[[self.polygon_id_col, 'geometry']])
            try:
                stats = self.get_modis_ndvi_evi_by_month(year, month, ee_polygons)
                size = stats.size().getInfo()

                if size == 0:
                    print(f"[✗] Chunk {idx} has no data, skipping.")
                    continue

                task = ee.batch.Export.table.toDrive(
                    collection=stats,
                    description=f"modis_y{year}_m{month:02d}_chunk{idx}",
                    folder=folder_name,
                    fileNamePrefix=f"modis_y{year}_m{month:02d}_chunk{idx}",
                    fileFormat='CSV'
                )
                task.start()
                print(f"[✓] Task submitted: {year}-{month:02d} | chunk {idx} | features: {size}")

            except Exception as e:
                print(f"[✗] Failed to export chunk {idx}: {e}")

    def export_all_months_modis(self,start_date, end_date, polygons, 
                                polygon_id_col="Ward"):
        year_months = self.extract_years_months(start_date, end_date)
        for year, month in year_months:
            self.export_modis_in_chunks(year, month, polygons, polygon_id_col=polygon_id_col)

    def download_from_drive_modis(self, output_folder='modis_exports', 
                                  drive_folder_name='modis_exports'):
        os.makedirs(output_folder, exist_ok=True)
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        # Get all folders named 'modis_exports'
        folder_list = drive.ListFile({
            'q': f"title = '{drive_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        }).GetList()

        if not folder_list:
            raise Exception(f"[✗] Folder '{drive_folder_name}' not found in Drive.")

        print(f"[✓] Found {len(folder_list)} folders named '{drive_folder_name}'.")

        all_files = []
        for folder in folder_list:
            folder_id = folder['id']
            query = f"'{folder_id}' in parents and trashed=false and title contains 'modis_y' and mimeType != 'application/vnd.google-apps.folder'"
            files = drive.ListFile({'q': query}).GetList()
            all_files.extend(files)

        print(f"[✓] Found {len(all_files)} total files across all '{drive_folder_name}' folders.")

        for file in tqdm(all_files, desc="Downloading from Drive"):
            file_path = os.path.join(output_folder, file['title'])
            try:
                file.GetContentFile(file_path)
            except Exception as e:
                print(f"[✗] Failed to download {file['title']}: {e}")

        for file in tqdm(all_files, desc="Deleting from Drive"):
            try:
                file.Delete()
            except Exception as e:
                print(f"[!] Could not delete {file['title']}: {e}")

        for folder in folder_list:
            try:
                folder.Delete()
                print(f"[✓] Deleted folder: {folder['title']} ({folder['id']})")
            except Exception as e:
                print(f"[!] Could not delete folder {folder['title']}: {e}")

    def combine_csv_exports_and_save_modis(self, csv_dir, 
                                           output_pkl='modis_ndvi_evi_stats.pkl'):
        all_csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
        print(f"Found {len(all_csvs)} CSV files...")

        df_list = []
        for file in tqdm(all_csvs, desc="Reading CSVs"):
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                os.remove(file)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_df.sort_values(by=['year', 'month'], inplace=True)
            final_df.to_pickle(output_pkl)
            print(f"[✓] Data saved to {output_pkl}")
            return final_df
        else:
            print("[✗] No CSVs to combine.")
            return pd.DataFrame()
        
    def run_full_modis_pipeline(self, start_date, end_date, polygons_gdf, 
                                polygon_id_col, output_path):
        self.export_all_months_modis(start_date, end_date, polygons_gdf, polygon_id_col)

        self.wait_for_gee_tasks_to_finish()

        print("[2] Downloading CSVs from Drive...")
        self.download_from_drive_modis()

        print("[3] Combining and saving final dataset...")
        df = self.combine_csv_exports_and_save_modis("modis_exports", output_pkl=output_path)

        shutil.rmtree('modis_exports', ignore_errors=True)
        print("[✓] MODIS pipeline complete.")
        return df

    def run_modis_pipeline(self):
            output_pkl = f"{self.OUTPUT}/modis_ndvi_evi_stats_{self._format_dates(self.start_date_evi, self.end_date_evi)}.pkl"
            
            return self.run_full_modis_pipeline(self.start_date_evi, 
                                                self.end_date_evi, self.polygons, 
                                                self.polygon_id_col, output_pkl)
    
#================================================
#--------------- EVI/NDVI (250m), by land use --------------------#
#--------------------------

    def get_clipped_land_use_class_image(self, year, month, ee_polygon, land_use_class):
        """
        Get a binary mask for the specified land use class:
        - Uses Dynamic World from June 2015 onwards
        - Falls back to MODIS MCD12Q1 for earlier dates
        Ensures correct resolution alignment (250m).
        """

        if (year > 2015) or (year == 2015 and month >= 6):
            # Use Dynamic World (10m → reproject to 250m)
            land_use_image = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                .filter(ee.Filter.calendarRange(year, year, 'year')) \
                .filter(ee.Filter.calendarRange(month, month, 'month')) \
                .select('label') \
                .map(lambda img: img.clip(ee_polygon)) \
                .reduce(ee.Reducer.mode()) \
                .eq(land_use_class)

            modis_ndvi_proj = ee.ImageCollection("MODIS/061/MOD13Q1") \
                .first() \
                .select('NDVI') \
                .projection()

            land_use_image = land_use_image.reproject(modis_ndvi_proj).setDefaultProjection(modis_ndvi_proj)

            return land_use_image

        # Fallback to MODIS MCD12Q1
        dw_to_modis_classes = {
            1: [1, 2, 3, 4, 5],     # trees
            2: [10],                # grass
            4: [12],                # crops
            5: [6, 7]               # shrub_and_scrub
        }

        modis_classes = dw_to_modis_classes.get(land_use_class)
        if modis_classes is None:
            raise ValueError(f"No MODIS match defined for DW class {land_use_class}")

        land_cover = ee.ImageCollection("MODIS/006/MCD12Q1") \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first() \
            .select('LC_Type1') \
            .clip(ee_polygon)

        modis_mask = land_cover.eq(modis_classes[0])
        for cls in modis_classes[1:]:
            modis_mask = modis_mask.Or(land_cover.eq(cls))

        #  Reproject to 250m resolution to match MOD13Q1 NDVI/EVI
        modis_ndvi_proj = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .first() \
            .select('NDVI') \
            .projection()

        modis_mask = modis_mask.reproject(modis_ndvi_proj).setDefaultProjection(modis_ndvi_proj)

        return modis_mask

    def get_clipped_modis_ndvi_evi(self, year, month, ee_polygon):
        """
        Get NDVI and EVI values from MODIS, clipped to each polygon.
        """
        modis = ee.ImageCollection("MODIS/061/MOD13Q1") \
                    .filter(ee.Filter.calendarRange(year, year, 'year')) \
                    .filter(ee.Filter.calendarRange(month, month, 'month')) \
                    .select(['NDVI', 'EVI']) \
                    .map(lambda img: img.clip(ee_polygon)) \
                    .mean() 
        
        return modis

    def process_chunk_with_reduceRegions_all_classes(self, year, month, polygons_chunk, 
                                                    scale=250, preview=False):
        """
        Use reduceRegions to process multiple polygons in one request, compute NDVI and EVI for all land use classes,
        and return results tagged with land use class information. Export all classes at once.
        
        Args:
        - year: The year for which the data is processed.
        - month: The month for which the data is processed.
        - polygons_chunk: The chunk of polygons to process.
        - scale: The scale to use for calculations.
        - preview: If True, process and display results for the first two polygons before processing all.
        """
        ee_polygon_collection = geemap.geopandas_to_ee(polygons_chunk)

        # If preview is enabled, limit to the first two polygons for a quick check
        if preview:
            ee_polygon_collection = ee_polygon_collection.limit(2)

        # Define land use classes and their names
        land_use_classes = {1: 'trees', 2: 'grass', 4: 'crops', 5: 'shrub_and_scrub'}
        
        all_class_results = []

        for class_value, class_name in land_use_classes.items():
            # Get the land use class image clipped for the polygons
            land_use_image = self.get_clipped_land_use_class_image(
                year, month, ee_polygon_collection.geometry(), class_value)

            if not land_use_image:
                continue

            # Get MODIS data clipped by land use class
            ndvi_evi_avg = self.get_clipped_modis_ndvi_evi(
                year, month, ee_polygon_collection.geometry()).updateMask(land_use_image)

            # Reduce the NDVI and EVI data by polygons
            reduced_result = ndvi_evi_avg.reduceRegions(
                collection=ee_polygon_collection,
                reducer=ee.Reducer.mean(),
                scale=scale
            )

            # Add land use class information to the reduced results
            class_results = reduced_result.map(lambda feature: feature.set('land_use_class', class_name))

            # Append results for this land use class
            all_class_results.append(class_results)

        # Merge all land use class results into one FeatureCollection
        all_results = ee.FeatureCollection(all_class_results).flatten()

        # Optionally preview the output before exporting
        if preview:
            print("Preview of NDVI/EVI data for the first two polygons (without geometry):")
            properties_only = all_results.map(
                lambda feature: ee.Feature(None, feature.toDictionary())
            ).getInfo()
            print(properties_only)

        return all_results  # Return FeatureCollection for export


    def export_chunk_to_drive(self, fc, description):
        """
        Export the processed chunk of results (all land use classes) to Google Drive.
        """
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            fileFormat='CSV',
            folder='MODIS_EVI_NDVI_land_use_full_month'  # Specify your Google Drive folder
        )
        task.start()

        print(f"Export task {description} started.")

    def process_data_in_chunks_EVI_land_use_parallel_all_classes(self,
            start_date, end_date, polygons, chunk_size=20, delay_between_chunks=5,
            delay_between_months=30, max_retries=3, scale=250):
        """
        Process NDVI/EVI for all land use classes using reduceRegions, in chunks,
        and wait for all exports to finish before proceeding to the next month.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        polygons = polygons.reset_index(drop=True)

        for date in date_range:
            year, month = date.year, date.month
            print(f"\n[🗓️] Processing EVI by land use {year}-{month:02d}")

            for attempt in range(max_retries):
                try:
                    for chunk_idx in range(0, len(polygons), chunk_size):
                        chunk = polygons.iloc[chunk_idx: chunk_idx + chunk_size]
                        print(f"Processing chunk {chunk_idx // chunk_size + 1} with {len(chunk)} polygons for {year}-{month:02d}")

                        fc = self.process_chunk_with_reduceRegions_all_classes(
                            year, month, chunk, scale=scale
                        )

                        description = f"Results_{year}_{month:02d}_chunk_{chunk_idx // chunk_size + 1}"
                        self.export_chunk_to_drive(fc, description)

                        # Wait between chunk submissions to avoid concurrency issues
                        time.sleep(delay_between_chunks)

                    # ✅ Wait until all chunk export tasks finish before continuing
                    print("[⏳] Waiting for GEE export tasks to finish before next month...")
                    self.wait_for_gee_tasks_to_finish()
                    break  # Success, break out of retry loop

                except Exception as e:
                    print(f"[!] Attempt {attempt+1} failed for {year}-{month:02d}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(10)
                    else:
                        print(f"[✗] Skipping {year}-{month:02d} after {max_retries} failed attempts.")

            # Optional delay before moving to next month
            time.sleep(delay_between_months)

    def load_processed_months(self,filename):
        """
        Load already processed months to avoid reprocessing.
        """
        if os.path.isfile(filename):
            df = pd.read_csv(filename, usecols=['year', 'month'])
            processed_months = set(tuple(x) for x in df[['year', 'month']].drop_duplicates().values)
        else:
            processed_months = set()
        
        return processed_months


    def retrieve_and_combine_evi_land_use_exports(self,
            output_folder='land_use_ndvi_evi_parallel',
            drive_folder_name='MODIS_EVI_NDVI_land_use_full_month',
            output_pkl='intermediary_datasets/land_use_ndvi_evi_stats_parallel.pkl',
            polygon_id_col='Ward',
            save_csv=False):

        self.wait_for_gee_tasks_to_finish()

        os.makedirs(output_folder, exist_ok=True)

        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        # Step 1: Locate all matching folders
        folder_list = drive.ListFile({
            'q': f"title = '{drive_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        }).GetList()

        if not folder_list:
            raise Exception(f"[✗] Folder '{drive_folder_name}' not found on Drive.")
        
        print(f"[✓] Found {len(folder_list)} folders named '{drive_folder_name}'.")

        df_list = []

        # Step 2: Search and download CSVs from each folder
        for folder in folder_list:
            folder_id = folder['id']
            files = drive.ListFile({
                'q': f"'{folder_id}' in parents and trashed = false and mimeType = 'text/csv'"
            }).GetList()

            print(f"[✓] Found {len(files)} CSVs in folder '{drive_folder_name}' ({folder_id})")

            for file in tqdm(files, desc=f"Downloading from folder {folder_id}"):
                title = file['title']
                match = re.search(r'(\d{4})_(\d{1,2})', title)
                if not match:
                    print(f"[!] Skipping file (can't parse date): {title}")
                    continue

                year, month = int(match.group(1)), int(match.group(2))
                path = os.path.join(output_folder, title)

                try:
                    file.GetContentFile(path)
                    df = pd.read_csv(path)
                    df['year'] = year
                    df['month'] = month
                    df_list.append(df)
                    file.Delete()  # delete after download
                except Exception as e:
                    print(f"[!] Failed to process {title}: {e}")

        # Step 3: Remove local copies and folder(s)
        for csv in glob.glob(os.path.join(output_folder, "*.csv")):
            os.remove(csv)
        shutil.rmtree(output_folder, ignore_errors=True)

        for folder in folder_list:
            try:
                folder.Delete()
                print(f"[🗑️] Deleted Drive folder '{folder['title']}' ({folder['id']})")
            except Exception as e:
                print(f"[!] Could not delete folder '{folder['title']}': {e}")

        # Step 4: Combine and save results
        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_df.sort_values(by=[polygon_id_col, 'year', 'month', 'land_use_class'], inplace=True)
            final_df.to_pickle(output_pkl)
            print(f"[💾] Saved combined file to {output_pkl}")

            if save_csv:
                csv_path = output_pkl.replace('.pkl', '.csv')
                final_df.to_csv(csv_path, index=False)
                print(f"[📄] Also saved CSV to {csv_path}")

            return final_df
        else:
            print("[✗] No dataframes to combine.")
            return pd.DataFrame()


    def run_land_use_ndvi_evi_pipeline(self, save_csv):
            output_pkl = f"{self.OUTPUT}/modis_evi_by_land_use_stats_{self._format_dates(self.start_date_evi, self.end_date_evi)}.pkl"
            self.process_data_in_chunks_EVI_land_use_parallel_all_classes(
                self.start_date_evi, self.end_date_evi, self.polygons, chunk_size=20, delay_between_chunks=5,
                  delay_between_months=60, max_retries=3, scale=250
            )
            return self.retrieve_and_combine_evi_land_use_exports(output_pkl=output_pkl, 
                                                                  polygon_id_col=self.polygon_id_col,
                                                                  save_csv=save_csv)



#================================================
#============================================================
#--------------- LAND USE (10m) --------------------#
#============================================================

    def get_land_use_area_percentage_all_classes(self, year, month, ee_polygon):
        """
        Get the percentage area of land use classes 0-7 within each polygon for the first 15 days of the month,
        excluding the snow and ice class (8) and the 'other' class (9).
        Uses Dynamic World from June 2015 onwards, MODIS MCD12Q1 before that.
        Adds 'land_cover_source' property to each feature.
        """
        if (year > 2015) or (year == 2015 and month >= 6):
            # Use Dynamic World (10m)
            land_use_mode = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                .filter(ee.Filter.calendarRange(year, year, 'year')) \
                .filter(ee.Filter.calendarRange(month, month, 'month')) \
                .select('label') \
                .map(lambda img: img.clip(ee_polygon)) \
                .reduce(ee.Reducer.mode())

            land_use_classes = {
                0: 'water',
                1: 'trees',
                2: 'grass',
                3: 'flooded_vegetation',
                4: 'crops',
                5: 'shrub_and_scrub',
                6: 'built',
                7: 'bare'
            }

            land_cover_source = 'DynamicWorld'

        else:
            # Use MODIS MCD12Q1 (500m)
            modis_land_cover = ee.ImageCollection("MODIS/006/MCD12Q1") \
                .filter(ee.Filter.calendarRange(year, year, 'year')) \
                .first() \
                .select('LC_Type1') \
                .clip(ee_polygon)

            dw_to_modis = {
                0: [0],
                1: [1, 2, 3, 4, 5],
                2: [10],
                3: [11],
                4: [12],
                5: [6, 7],
                6: [13],
                7: [16]
            }

            land_use_classes = {
                0: 'water',
                1: 'trees',
                2: 'grass',
                3: 'flooded_vegetation',
                4: 'crops',
                5: 'shrub_and_scrub',
                6: 'built',
                7: 'bare'
            }

            # Remap MODIS LC_Type1 to DW class equivalents
            land_use_mode = ee.Image(0).where(modis_land_cover.eq(-9999), -1)
            for dw_class, modis_classes in dw_to_modis.items():
                mask = modis_land_cover.eq(modis_classes[0])
                for cls in modis_classes[1:]:
                    mask = mask.Or(modis_land_cover.eq(cls))
                land_use_mode = land_use_mode.where(mask, dw_class)

            land_cover_source = 'MODIS'

        land_use_areas = {}

        for class_value in land_use_classes.keys():
            if land_cover_source == 'DynamicWorld':
                mask = land_use_mode.eq(class_value)
            else:
                modis_classes = dw_to_modis.get(class_value, [])
                mask = land_use_mode.eq(modis_classes[0])
                for cls in modis_classes[1:]:
                    mask = mask.Or(land_use_mode.eq(cls))

            land_use_areas[class_value] = mask.multiply(ee.Image.pixelArea())

        return land_use_areas, land_use_classes, land_cover_source


    def process_chunk_with_reduceRegions_land_use_percentage_all_classes(self, year, month, 
                                                                        polygons_chunk, scale=250, 
                                                                        preview=False):
        """
        Use reduceRegions to process multiple polygons in one request, compute percentage area for all land use classes,
        and return results tagged with land use class and land cover source information. Export all classes at once.
        """
        ee_polygon_collection = geemap.geopandas_to_ee(polygons_chunk)

        # Get the area for all land use classes and their names for the given year and month
        land_use_areas, land_use_classes, land_cover_source = self.get_land_use_area_percentage_all_classes(
            year, month, ee_polygon_collection.geometry()
        )
        
        all_class_results = []

        # Calculate the area of each polygon directly using its geometry and set it as a property
        ee_polygon_collection = ee_polygon_collection.map(
            lambda feature: feature.set('total_area', feature.geometry().area())
        )

        # Process each land use class area and calculate percentage
        for class_value, land_use_area in land_use_areas.items():
            class_name = land_use_classes[class_value]
            
            # Calculate the area of the land use class for each polygon
            land_use_area_sum = land_use_area.reduceRegions(
                collection=ee_polygon_collection,
                reducer=ee.Reducer.sum(),
                scale=scale
            ).map(lambda feature: feature.set(
                'land_use_class', class_value,
                'land_use_name', class_name,
                'land_use_area', feature.get('sum'),
                'land_cover_source', land_cover_source
            ))

            # Calculate percentage using properties within Earth Engine
            percentage_results = land_use_area_sum.map(
                lambda feature: feature.set(
                    'land_use_percentage',
                    ee.Number(feature.get('land_use_area'))
                    .divide(ee.Number(feature.get('total_area')))
                    .multiply(100)
                )
            )

            # Append results for this land use class
            all_class_results.append(percentage_results)

        # Merge all land use class results into one FeatureCollection
        all_results = ee.FeatureCollection(all_class_results).flatten()

        # Optionally preview the output before exporting
        if preview:
            print("Preview of land use percentages for the first two polygons (without geometry):")
            properties_only = all_results.limit(5).map(
                lambda feature: ee.Feature(None, feature.toDictionary())
            ).getInfo()
            print(properties_only)

        return all_results  # Return FeatureCollection for export

    def export_chunk_to_drive_land_use(self, fc, description):
        """
        Export the processed chunk of results (all land use classes) to Google Drive.
        """
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            fileFormat='CSV',
            folder='Land_Use_Percentage'  # Specify your Google Drive folder
        )
        task.start()

        print(f"Export task {description} started.")

    def process_data_in_chunks_land_use_percentage_parallel_all_classes(self, start_date, end_date, polygons,
                                                                        chunk_size, delay_between_chunks, 
                                                                        delay_between_months, max_retries, scale=250):
        """
        Process data in chunks using reduceRegions to handle multiple polygons at once, compute percentage area 
        for all land use classes, and export results to Google Drive. Handles throttling to avoid EE rate limits.
        """
        year_month_pairs = self.extract_years_months(start_date, end_date)
        polygon_chunks = [polygons.iloc[i:i + chunk_size] for i in range(0, len(polygons), chunk_size)]
        processed_modis_years = set()

        EXPORTS_BEFORE_PAUSE = 3           # Submit 3 exports before pausing
        EXPORT_PAUSE_SECONDS = 120         # Wait 2 minutes between batches

        with tqdm(total=len(year_month_pairs), desc="Processing months") as pbar_months:
            for year, month in year_month_pairs:
                if (year < 2015) or (year == 2015 and month < 6):
                    if year in processed_modis_years:
                        print(f"[↩️] Skipping {year}-{month}: already processed MODIS land cover for this year.")
                        pbar_months.update(1)
                        continue
                    else:
                        processed_modis_years.add(year)
                        use_modis = True
                else:
                    use_modis = False

                retries = 0
                delay_multiplier = 1

                while retries < max_retries:
                    try:
                        with tqdm(total=len(polygon_chunks), desc=f"Processing land use chunks for {year}-{month}", leave=False) as pbar_chunks:
                            for i, polygons_chunk in enumerate(polygon_chunks):
                                print(f"Processing chunk {i+1} with {len(polygons_chunk)} polygons for {year}-{month}")

                                fc = self.process_chunk_with_reduceRegions_land_use_percentage_all_classes(
                                    year, month, polygons_chunk, scale)

                                if use_modis:
                                    export_name = f"Land_Use_Percentage_{year}_chunk_{i+1}"
                                else:
                                    export_name = f"Land_Use_Percentage_{year}_{month}_chunk_{i+1}"

                                self.export_chunk_to_drive_land_use(fc, export_name)

                                pbar_chunks.update(1)
                                time.sleep(delay_between_chunks)

                                # 🔁 Pause every N exports to prevent quota breach
                                if (i + 1) % EXPORTS_BEFORE_PAUSE == 0:
                                    print(f"[⏸️] Pausing {EXPORT_PAUSE_SECONDS}s after {EXPORTS_BEFORE_PAUSE} exports...")
                                    time.sleep(EXPORT_PAUSE_SECONDS)

                        print("[⏳] Waiting for GEE export tasks to finish before next month...")
                        self.wait_for_gee_tasks_to_finish()

                        pbar_months.update(1)
                        time.sleep(delay_between_months * delay_multiplier)
                        break

                    except ee.EEException as e:
                        print(f"[⚠️] Error for {year}-{month}: {str(e)}. Retrying in {delay_between_months * delay_multiplier} seconds...")
                        retries += 1
                        time.sleep(delay_between_months * delay_multiplier)
                        delay_multiplier *= 2

                        if retries == max_retries:
                            print(f"[✗] Max retries reached for {year}-{month}. Skipping.")
                            pbar_months.update(1)
                            break

    def retrieve_and_combine_land_use_percentage_exports(self, output_folder='land_use_pct_exports',
                                                        drive_folder_name='Land_Use_Percentage',
                                                        output_pkl='intermediary_datasets/land_use_area_percentage_parallel.pkl',
                                                        polygon_id_col='Ward',
                                                        save_csv=False):
        """
        Waits for tasks, downloads from Drive, parses year/month from filenames, combines all CSVs,
        deletes local and Drive folders, and saves to a pickle (and optionally CSV).
        Supports both monthly (Dynamic World) and yearly (MODIS) exports.
        """
        # Step 1: Wait for EE tasks
        self.wait_for_gee_tasks_to_finish()

        # Step 2: Auth and setup
        os.makedirs(output_folder, exist_ok=True)

        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        # Step 3: Locate folder
        folder_list = drive.ListFile({
            'q': f"title contains '{drive_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        }).GetList()

        if not folder_list:
            raise Exception(f"[✗] No folders starting with '{drive_folder_name}' found.")

        df_list = []

        for folder in folder_list:
            folder_id = folder['id']
            print(f"[📂] Processing Drive folder: {folder['title']}")

            files = drive.ListFile({
                'q': f"'{folder_id}' in parents and trashed = false and mimeType = 'text/csv'"
            }).GetList()

            for file in tqdm(files, desc=f"Downloading from {folder['title']}"):
                title = file['title']
                match = re.search(r'(\d{4})(?:_(\d{1,2}))?', title)
                if not match:
                    print(f"[!] Skipping: could not parse date from {title}")
                    continue

                year = int(match.group(1))
                month = int(match.group(2)) if match.group(2) else pd.NA
                local_path = os.path.join(output_folder, title)

                try:
                    file.GetContentFile(local_path)
                    df = pd.read_csv(local_path)
                    df['year'] = year
                    df['month'] = month
                    df_list.append(df)
                except Exception as e:
                    print(f"[!] Failed to process {title}: {e}")


        # Step 4: Find CSVs
        files = drive.ListFile({
            'q': f"'{folder_id}' in parents and trashed = false and mimeType = 'text/csv'"
        }).GetList()
        print(f"[✓] Found {len(files)} CSV files in Drive folder '{drive_folder_name}'")

        df_list = []

        for file in tqdm(files, desc="Downloading and parsing"):
            title = file['title']
            match = re.search(r'(\d{4})(?:_(\d{1,2}))?', title)  # Match year and optional month
            if not match:
                print(f"[!] Skipping: could not parse date from {title}")
                continue

            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else pd.NA  # pd.NA if month is missing

            local_path = os.path.join(output_folder, title)

            try:
                file.GetContentFile(local_path)
                df = pd.read_csv(local_path)
                df['year'] = year
                df['month'] = month
                df_list.append(df)
            except Exception as e:
                print(f"[!] Failed to process {title}: {e}")

        # Step 5: Cleanup local files
        for csv in glob.glob(os.path.join(output_folder, "*.csv")):
            os.remove(csv)
        shutil.rmtree(output_folder, ignore_errors=True)

        # Step 6: Delete Drive folder
        try:
            drive_folder = drive.CreateFile({'id': folder_id})
            drive_folder.Delete()
            print(f"[🗑️] Deleted Drive folder '{drive_folder_name}'")
        except Exception as e:
            print(f"[!] Could not delete Drive folder: {e}")

        # Step 7: Combine and save
        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_df.sort_values(by=[self.polygon_id_col, 'year', 'month', 'land_use_class'], inplace=True)
            final_df.to_pickle(output_pkl)
            print(f"[💾] Saved combined dataset to {output_pkl}")

            if save_csv:
                csv_path = output_pkl.replace('.pkl', '.csv')
                final_df.to_csv(csv_path, index=False)
                print(f"[📄] Also saved CSV to {csv_path}")

            return final_df
        else:
            print("[✗] No dataframes to combine.")
            return pd.DataFrame()


    def run_land_use_percentage_pipeline(self, save_csv):
            output_pkl = f"{self.OUTPUT}/land_use_pct_stats_{self._format_dates(self.start_date_evi, self.end_date_evi)}.pkl"
            self.process_data_in_chunks_land_use_percentage_parallel_all_classes(
                self.start_date_evi, self.end_date_evi, self.polygons, chunk_size= 20,
                  delay_between_chunks=5,delay_between_months= 60,max_retries= 3,scale= 250
            )
            return self.retrieve_and_combine_land_use_percentage_exports(output_pkl=output_pkl, 
                                                                         polygon_id_col=self.polygon_id_col,
                                                                         save_csv=save_csv)

#============================================================
#--------------- CONFLICT (ACLED) --------------------#
#============================================================
    def run_conflict_pipeline(self, api_key_acled, email_acled, country):
        import requests
        from shapely.geometry import Point

        # Define the API endpoint
        url = "https://api.acleddata.com/acled/read"
        
        # ---------------------------------------------
        # 1) Compute adjusted start year for ACLED data
        # ---------------------------------------------
        start_date_evi = pd.to_datetime(self.start_date_evi)
        end_date_evi = pd.to_datetime(self.end_date_evi)

        # Take 12 months back
        start_date_acled = start_date_evi - pd.DateOffset(months=12)

        year_start = start_date_acled.year
        year_end = end_date_evi.year + 1  # include full end year

        print(f"Fetching ACLED data from {year_start} to {year_end-1} to ensure 12 months of history.")

        # ---------------------------------------------
        # 2) Fetch ACLED data for expanded range
        # ---------------------------------------------
        years = range(year_start, year_end)
        data_frames = []

        for year in years:
            params = {
                "key": api_key_acled,
                "email": email_acled,
                "country": country,
                "year": year,
                "format": "json"
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df_year = pd.DataFrame(data['data'])
                    data_frames.append(df_year)
                    print(f"Data for year {year} retrieved successfully.")
                else:
                    print(f"No data found for year {year}")
            else:
                print(f"Error fetching data for year {year}: {response.status_code}, Full response: {response.json()}")

        # Combine all yearly data
        conflict_data = pd.concat(data_frames, ignore_index=True)
        
        conflict_data['fatalities'] = pd.to_numeric(conflict_data['fatalities'], 
                                                    errors='coerce').fillna(0).astype(int)

        # -------------------------------------------------------------
        # Calculate 12-month rolling conflict metrics
        # -------------------------------------------------------------
        date_range = pd.date_range(start=self.start_date_evi, end=self.end_date_evi, freq="MS")
        date_df = pd.DataFrame({'date': date_range})
        date_df['year'] = date_df['date'].dt.year
        date_df['month'] = date_df['date'].dt.month

        conflict_data['event_date'] = pd.to_datetime(conflict_data['event_date'], errors='coerce')
        conflict_data = conflict_data.dropna(subset=['event_date'])  # drop invalid dates
        conflict_data['year'] = conflict_data['event_date'].dt.year
        conflict_data['month'] = conflict_data['event_date'].dt.month

        # Filter for violent conflict types
        violent_conflict_types = [
            "Battles", "Explosions/Remote violence",
            "Violence against civilians", "Riots"
        ]
        conflict_data = conflict_data[conflict_data['event_type'].isin(violent_conflict_types)]

        # Aggregate
        conflict_data['conflict_total_num'] = conflict_data.groupby(
            ['latitude', 'longitude', 'year', 'month'])['event_id_cnty'].transform('count')
        conflict_data['conflict_fatalities_num'] = conflict_data.groupby(
            ['latitude', 'longitude', 'year', 'month'])['fatalities'].transform('sum')
        conflict_data = conflict_data.drop_duplicates(subset=['latitude', 'longitude', 'year', 'month'])

        conflict_data['geometry'] = conflict_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        conflict_gdf = gpd.GeoDataFrame(conflict_data, geometry='geometry', crs="EPSG:4326")

        wards_NDMA_projected = self.polygons.to_crs("EPSG:21096")
        conflict_gdf_proj = conflict_gdf.to_crs("EPSG:21096")

        distance_cutoffs = [500000, 100000]
        for distance_cutoff in distance_cutoffs:
            results = []
            for _, polygon in wards_NDMA_projected.iterrows():
                centroid = polygon.geometry.centroid
                conflict_gdf_proj['distance_to_polygon'] = conflict_gdf_proj.geometry.distance(centroid)
                conflict_within = conflict_gdf_proj[conflict_gdf_proj['distance_to_polygon'] <= distance_cutoff].copy()

                for col in ['conflict_total_num', 'conflict_fatalities_num']:
                    conflict_within[col] = pd.to_numeric(conflict_within[col], errors='coerce')

                conflict_within['conflict_total_num_w'] = conflict_within['conflict_total_num'] / np.log(conflict_within['distance_to_polygon'] + 1)
                conflict_within['conflict_fatalities_num_w'] = conflict_within['conflict_fatalities_num'] / np.log(conflict_within['distance_to_polygon'] + 1)

                monthly_totals = conflict_within.groupby(['year', 'month']).agg({
                    'conflict_total_num_w': 'sum',
                    'conflict_fatalities_num_w': 'sum'
                }).reset_index().rename(columns={
                    'conflict_total_num_w': 'monthly_conflicts_dis_w',
                    'conflict_fatalities_num_w': 'monthly_fatalities_dis_w'
                })

                monthly_totals = pd.merge(date_df, monthly_totals, on=['year', 'month'], how='left')
                monthly_totals.fillna(0, inplace=True)

                monthly_totals['conflict_previous_12m'] = monthly_totals['monthly_conflicts_dis_w'].rolling(12, min_periods=1).sum()
                monthly_totals['fatalities_previous_12m'] = monthly_totals['monthly_fatalities_dis_w'].rolling(12, min_periods=1).sum()
                monthly_totals[self.polygon_id_col] = polygon[self.polygon_id_col]
                results.append(monthly_totals)

            final_df = pd.concat(results, ignore_index=True)
            final_df = final_df[[self.polygon_id_col, 'year', 'month', 'monthly_conflicts_dis_w', 
                                'monthly_fatalities_dis_w', 'conflict_previous_12m', 'fatalities_previous_12m']]

            distance_cutoff_km = int(distance_cutoff / 1000)

            date_range = pd.date_range(start=self.start_date_evi, end=self.end_date_evi, freq="MS")
            start_label = date_range.min().strftime('%b_%Y')
            end_label = date_range.max().strftime('%b_%Y')

            if start_label == end_label:
                date_label = start_label
            else:
                date_label = f"{start_label}_{end_label}"

            final_df.to_csv(os.path.join(
                OUTPUT,
                f"ACLED_conflict_12m_running_sum_by_polygon_{distance_cutoff_km}kmcutoff_{date_label}.csv"
            ), index=False)

#============================================================
#--------------- POPULATION DENSITY --------------------#
#============================================================

    def compute_population_density_changes(self, fc):
        years = self.years_pop_den
        pop_density_ic = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")

        images = {
            year: pop_density_ic.filterDate(f'{year}-01-01', f'{year}-12-31').first().select('population_density')
            for year in years
        }

        def map_density(feature):
            for i in range(1, len(years)):
                prev_year, curr_year = years[i - 1], years[i]
                prev = images[prev_year].reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=feature.geometry(), scale=927.67).get('population_density')
                curr = images[curr_year].reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=feature.geometry(), scale=927.67).get('population_density')
                delta = ee.Number(curr).subtract(prev)

                feature = feature.set({
                    f'density_{prev_year}': prev,
                    f'density_{curr_year}': curr,
                    f'delta_{prev_year}_{curr_year}': delta
                })

            return feature

        return fc.map(map_density)


    def run_population_density_change_pipeline(self, polygons_gdf, output_csv, chunk_size=50):
        data = []
        years = self.years_pop_den
        num_chunks = int(np.ceil(len(polygons_gdf) / chunk_size))

        for i in range(num_chunks):
            print(f"Processing chunk {i+1}/{num_chunks}")
            chunk = polygons_gdf.iloc[i*chunk_size:(i+1)*chunk_size]
            chunk_fc = geemap.geopandas_to_ee(chunk)

            try:
                result_chunk = self.compute_population_density_changes(chunk_fc)
                result_dict = result_chunk.getInfo()

                features = result_dict['features']

                # Dynamic extraction
                chunk_data = []
                for f in features:
                    props = f['properties']
                    row = {self.polygon_id_col: props.get(self.polygon_id_col)}

                    # Add densities
                    for year in years:
                        row[f'density_{year}'] = props.get(f'density_{year}')

                    # Add deltas
                    for j in range(1, len(years)):
                        prev_year = years[j-1]
                        curr_year = years[j]
                        row[f'delta_{prev_year}_{curr_year}'] = props.get(f'delta_{prev_year}_{curr_year}')

                    chunk_data.append(row)

                data.extend(chunk_data)

            except Exception as e:
                print(f"[!] Error in chunk {i+1}: {e}")

        pop_den_df = pd.DataFrame(data)
        print(pop_den_df.head())
        pop_den_df.to_csv(output_csv, index=False)
        print(f"[✓] Exported population density changes to {output_csv}")
        return pop_den_df


    def run_population_pipeline(self):
                    output_csv = os.path.join(self.OUTPUT, "population_density_2005_2020.csv")
                    return self.run_population_density_change_pipeline(self.polygons, output_csv)

#============================================================
#--------------- ACCESSIBILITY/REMOTENESS --------------------#
#============================================================

# Load the accessibility images
    def run_accessibility_to_cities_pipeline(self, polygons_gdf, output_csv, chunk_size=50):
        accessibility_to_cities_2015 = ee.Image("Oxford/MAP/accessibility_to_cities_2015_v1_0").select('accessibility')
        
        data = []
        num_chunks = int(np.ceil(len(polygons_gdf) / chunk_size))

        for i in range(num_chunks):
            print(f"Processing chunk {i+1}/{num_chunks}")
            chunk = polygons_gdf.iloc[i*chunk_size:(i+1)*chunk_size]
            chunk_fc = geemap.geopandas_to_ee(chunk)

            def map_accessibility(feature):
                city_access = accessibility_to_cities_2015.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=feature.geometry(),
                    scale=927.67
                ).get('accessibility')
                return feature.set({
                    'accessibility_to_cities_2015': city_access,
                    'Ward': feature.get('Ward'),
                    polygon_id_col: feature.get(self.polygon_id_col)
                })

            try:
                result = chunk_fc.map(map_accessibility).getInfo()
                features = result['features']
                chunk_data = [{
                    polygon_id_col: f['properties'].get('Ward') or f['properties'].get(self.polygon_id_col),
                    'travel_time_to_cities_2015': f['properties'].get('accessibility_to_cities_2015')
                } for f in features]
                data.extend(chunk_data)
            except Exception as e:
                print(f"[!] Error in chunk {i+1}: {e}")

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"[✓] Saved to: {output_csv}")
        return df

    
    def run_accessibility_pipeline(self):
                    output_csv = os.path.join(self.OUTPUT, "accessibility_to_cities_2015.csv")
                    return self.run_accessibility_to_cities_pipeline(self.polygons, output_csv)
#============================================================
                #----------------- Helper -----------------#
    def _format_dates(self, start, end):
                    start_str = datetime.strptime(start, '%Y-%m-%d').strftime('%Y_%m')
                    end_str = datetime.strptime(end, '%Y-%m-%d').strftime('%Y_%m')
                    return f"{start_str}_to_{end_str}"



#----------------

def main():
    try:
        parser = argparse.ArgumentParser(description='Generate geospatial variables')
        parser.add_argument("--start_date", required=True)
        parser.add_argument("--end_date", required=True)
        parser.add_argument("--pickle_file", required=True)
        parser.add_argument("--polygon_id", default="Ward")
        parser.add_argument("--pop_years", default="2015,2020")
        parser.add_argument("--api_key_acled")
        parser.add_argument("--email_acled")
        parser.add_argument("--country", default="Kenya")
        args = parser.parse_args()

        # Load data
        muac = gpd.GeoDataFrame(pd.read_pickle(args.pickle_file))
        polygons = gpd.read_file(os.path.join(SHAPE, 'Kenya_wards_NDMA.shp'))

        # Visualization
        polygons.plot(color='lightblue', edgecolor='black')
        plt.title(f"{args.polygon_id} in dataset")
        plt.show()

        # Change directory
        os.chdir('/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation')

        # Initialize generator
        generator = SecondaryVariableGenerator(
            shape_path=SHAPE,
            output_path=OUTPUT,
            polygon_id_col=args.polygon_id,
            start_date_main=args.start_date,
            end_date_main=args.end_date,
            start_date_evi=args.start_date,
            end_date_evi=args.end_date,
            polygons=polygons,
            years_pop_den=[int(y) for y in args.pop_years.split(",")]
        )

        # Execute pipelines
        print("\n=== Starting Geospatial Variable Generation ===")
        
        pipelines = [
            ("CHIRPS precipitation", generator.run_chirps_pipeline),
            ("ERA5 temperature", generator.run_era5_pipeline),
            ("MODIS vegetation indices", generator.run_modis_pipeline),
            ("Land-use vegetation indices", lambda: generator.run_land_use_ndvi_evi_pipeline(save_csv=False)),
            ("Land use percentage", lambda: generator.run_land_use_percentage_pipeline(save_csv=False)),
            ("Population density", generator.run_population_pipeline),
            ("Accessibility", generator.run_accessibility_pipeline)
        ]

        for i, (name, pipeline) in enumerate(pipelines, 1):
            print(f"\n[{i}/{len(pipelines)}] Running {name} pipeline...")
            pipeline()

        # Handle conflict pipeline separately
        if args.api_key_acled and args.email_acled:
            print(f"\n[{len(pipelines)+1}/{len(pipelines)+1}] Running conflict data pipeline...")
            generator.run_conflict_pipeline(
                api_key_acled=args.api_key_acled,
                email_acled=args.email_acled,
                country=args.country
            )
        else:
            print("\n[!] Skipping conflict data pipeline (missing API credentials)")

        print("\n=== All geospatial pipelines completed successfully ===")

    except Exception as e:
        print(f"\n❌ Error in geospatial processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()

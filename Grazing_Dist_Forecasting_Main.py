import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from sqlalchemy import create_engine, text
from Grazing_Dist_Forecast_Model import run_precip_forecast_pipeline
from Grazing_Dist_Forecast_Model import GrazingDistForecaster


def process_grazing_distance_forecasts(county_id):
    # Create SQLAlchemy engine
    engine = create_engine(
        'mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones'
        # 'mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones'    
    )

    query = """
        SELECT hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId as qid, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
                DATE(data_collection_exercise.ExerciseStartDate) as ExerciseStartDate, Sum(hh_livestock_milk_production_per_species.DailyQntyMilkedInLtrs) as amountmilked,Sum(hh_livestock_milk_production_per_species.DailyQntyConsumedInLtrs) as amountconsumed,Sum(hh_livestock_milk_production_per_species.DailyQntySoldInLtrs) as amountsold, Sum(hh_livestock_milk_production_per_species.PricePerLtr) as PricePerLtr,wards.WardName as Shapefile_wardName
        FROM (hh_livestock_milk_production_per_species
            LEFT JOIN hha_questionnaire_sessions ON (hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
            LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)LEFT JOIN wards ON (hha_questionnaire_sessions.WardId = wards.WardId)
        WHERE (hha_questionnaire_sessions.CountyId = %s)
        GROUP BY hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,data_collection_exercise.ExerciseStartDate, wards.Shapefile_wardName
    """

    #db_df1 = pd.read_sql(query, conn)
    db_df1 = pd.read_sql(query, engine, params=(county_id,))

    query = """
        SELECT kia_questionnaire_sessions.KiaQuestionnaireSessionId as qid, kia_questionnaire_sessions.CountyId, kia_questionnaire_sessions.WardId, kia_questionnaire_sessions.SubCountyId,
                DATE(data_collection_exercise.ExerciseStartDate) as ExerciseStartDate, avg(kia_water_resources.DistInKmsToWaterSourceFromGrazingArea) AS GrazingDist, avg(kia_water_resources.DistInKmsToWaterSourceForHouseholds) AS WaterDist, kia_water_resources.KiaQuestionnaireSessionId,data_collection_exercise.DataCollectionExerciseId 
        FROM (kia_water_resources
            LEFT JOIN kia_questionnaire_sessions ON (kia_water_resources.KiaQuestionnaireSessionId = kia_questionnaire_sessions.KiaQuestionnaireSessionId))
            LEFT JOIN data_collection_exercise ON (kia_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)LEFT JOIN wards ON (kia_questionnaire_sessions.WardId = wards.WardId) 
        WHERE (kia_questionnaire_sessions.CountyId = %s)
        GROUP BY kia_questionnaire_sessions.CountyId, kia_questionnaire_sessions.WardId, kia_questionnaire_sessions.SubCountyId,data_collection_exercise.ExerciseStartDate, wards.Shapefile_wardName,kia_water_resources.KiaQuestionnaireSessionId
    """

    #db_df1 = pd.read_sql(query, conn)
    db_df2 = pd.read_sql(query, engine, params=(county_id,))

    db_df2=db_df2.groupby(['WardId','ExerciseStartDate'])[['GrazingDist']].mean().reset_index()
    #db_df2

    db_df3= pd.merge(db_df1, db_df2, left_on=['WardId', 'ExerciseStartDate'], right_on=['WardId', 'ExerciseStartDate'], how='left')
    #db_df3

    db_df3['ExerciseStartDate'] = pd.to_datetime(db_df3['ExerciseStartDate'])    

    query = """
    SELECT 
        Seasons.season,
        Seasons.Season_Index,
        Seasons.Month,
        DATE_FORMAT(STR_TO_DATE(CONCAT(LEFT(LTAs.Month, 3), ' 1 2000'), '%b %d %Y'), '%M') AS LTAMonth,
        LTAs.Bad_year,
        LTAs.Good_year
    FROM Seasons
    LEFT JOIN LTAs ON (Seasons.month = DATE_FORMAT(STR_TO_DATE(CONCAT(LEFT(LTAs.Month, 3), ' 1 2000'), '%b %d %Y'), '%M'))
    WHERE (LTAs.CountyId = %s AND LTAs.Indicator='Milk Production')
    """

    #Seasons = pd.read_sql(query, conn)
    Seasons = pd.read_sql(query, engine, params=(county_id,))


    db_df3['year'] = db_df3['ExerciseStartDate'].dt.year
    db_df3['month'] = db_df3['ExerciseStartDate'].dt.strftime('%B') 
    db_df3['month_num'] = db_df3['ExerciseStartDate'].dt.month

    db_df = db_df3.merge(Seasons, left_on=['month'], right_on=['Month'], how='right')

    #conn.close()
    #db_df

    '''conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='Romans17:48',
            database='livelihoodzones_5'
        )

    cursor = conn.cursor()
    '''

    query = """
        SELECT *
        FROM Precipitation LEFT JOIN counties ON (counties.CountyName = Precipitation.COUNTY)
        WHERE (counties.CountyId = %s)
        """

    #precipitation_df = pd.read_sql(query, conn)
    precipitation_df = pd.read_sql(query, engine, params=(county_id,))



    prep_df0 = precipitation_df.groupby(['WARD','T'])['precipitation'].sum()
    prep_df0 = prep_df0.reset_index()

    #conn.close()
    prep_df0['T'] = pd.to_datetime(prep_df0['T'], errors='coerce')
    prep_df0['year'] = prep_df0['T'].dt.year
    prep_df0['month_name'] = prep_df0['T'].dt.strftime('%B') 
    prep_df0['month_num'] = prep_df0['T'].dt.month
    prep_df0
    prep_df0 = Seasons.merge(prep_df0, left_on=['Month'], right_on=['month_name'], how='right')
    #prep_df0
    #Wusi/Kishamba Chala
    #unique_wards = prep_df0["WARD"].unique()
    unique_wards = db_df1[db_df1['Shapefile_wardName'].notna()]
    unique_wards = unique_wards['Shapefile_wardName'].unique()
    #unique_wards = ['Chala']
    for WARD in unique_wards:
        print(f"Processing {WARD}...")
        prep_df = prep_df0[prep_df0["WARD"] == WARD]
        prep_df = prep_df.reset_index()    
        prep_df=prep_df[['season','Season_Index','Month','WARD','T','precipitation','year','month_name','month_num']]
        MIN_REQUIRED_ROWS = 48  # seq_length + 1
        if prep_df.shape[0] < MIN_REQUIRED_ROWS:    
            # print(f"No precipitation data found for {WARD}. Skipping...")
            print(f"Insufficient precipitation data for {WARD} (only {prep_df.shape[0]} rows). Skipping...")        
            continue                       
        unique_ward = prep_df["WARD"].unique()
        #prep_df1=prep_df
        #prep_df['T'] = pd.to_datetime(prep_df['T'])
        #pd.to_datetime(prep_df0['T'], errors='coerce')
        # cutoff_date = pd.to_datetime('2025-03-01')
        # prep_df=prep_df[(prep_df['T'] < cutoff_date)]

        # Call the precipitation forecasting function
        results = run_precip_forecast_pipeline(
            prep_df=prep_df,
            seq_length=48,
            forecast_start_year=2016,
            forecast_start_month=1,
            n_future=109
        )

        # Access the results
        model = results["model"]
        scaler = results["scaler"]
        forecast_df = results["forecast_df"]
        test_metrics = results["test_metrics"]

        print(f"RMSE: {test_metrics.get('rmse')}")
        print(f"Forecast periods with actuals: {results['forecast_metrics'].get('n_with_actuals', 0)}")

        if not forecast_df.empty:
            print(f"Forecast first 5 periods: \n{forecast_df[['Date', 'Forecasted Precipitation']].head()}")

        print(forecast_df.head())

        forecast_df1=forecast_df[['Month','Year','Date_Object','Forecasted Precipitation','Forecast Uncertainty (Std Dev)']]
        prep_df1 = forecast_df1.merge(prep_df, left_on=['Month','Year','Date_Object'], right_on=['month_num','year','T'], how='right')
        prep_df2=prep_df1[['year','Forecasted Precipitation','WARD','T','precipitation','month_name','month_num']]
        precipitation_forecasts_df=prep_df2
        db_df=db_df[['WardId','HouseHoldId','Shapefile_wardName', 'month', 'year', 'season','Season_Index','amountmilked','GrazingDist','Bad_year','Good_year']]


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

            if 'year_month' not in db_df.columns:
                db_df_clean['year_month'] = db_df_clean['year'].astype(str) + '-' + db_df_clean['month'].astype(str)
            
            for variable in variables:
                print(f"\nReplacing outliers for {variable} with ward-month averages")
                outlier_col = f'{variable}_is_outlier'
                
                # Check if outlier column exists
                if outlier_col not in db_df.columns:
                    print(f"Warning: No outlier data found for {variable}, skipping replacement")
                    continue

                for ward in db_df_clean['ward'].unique():
                    for year_month in db_df_clean['year_month'].unique():

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

                            ward_avg = db_df_clean.loc[(db_df_clean['ward'] == ward) & ~db_df_clean[outlier_col], variable].mean()
                            if np.isnan(ward_avg):
                                print(f"Cannot find suitable replacement for ward {ward}, {year_month}. Keeping original values.")
                                continue
                            replacement_value = ward_avg
                        else:

                            non_outlier_mean = subset.loc[~subset[outlier_col], variable].mean()
                            replacement_value = non_outlier_mean

                        outlier_mask = mask & db_df_clean[outlier_col]
                        if outlier_mask.any():
                            db_df_clean.loc[outlier_mask, variable] = replacement_value
                            print(f"Replaced {outlier_mask.sum()} outliers in ward {ward}, {year_month} for {variable}")
            
            return db_df_clean

        db_df_clean = replace_outliers_with_averages(db_df, variables=['amountmilked'])
        db_df_clean

        unique_ward
        unique_ward1 = unique_ward[0]
        unique_ward2 = unique_ward[0]
        unique_ward1


            # db_df_clean1=db_df_clean.groupby(['Shapefile_wardName', 'month', 'year', 'season','Season_Index','Bad_year','Good_year'])[['amountmilked']].mean().reset_index()
        db_df_clean1=db_df_clean.groupby(['Shapefile_wardName', 'month', 'year', 'season','Season_Index','Bad_year','Good_year'])[['amountmilked','GrazingDist']].mean().reset_index()

        joined_data2 = db_df_clean1.merge(prep_df2, left_on=['Shapefile_wardName', 'year', 'month'], right_on=['WARD', 'year', 'month_name'], how='right')

        joined_data3=joined_data2[(joined_data2['Shapefile_wardName']==unique_ward1)&(joined_data2['year']>2016)]

        MIN_REQUIRED_ROWS = 14  # seq_length + 1

        if joined_data3.shape[0] < MIN_REQUIRED_ROWS:
            print(f"Insufficient data for {WARD} (only {joined_data3.shape[0]} rows). Skipping...")
            continue     

        data_numeric = joined_data3.assign(**{col: joined_data3[col].map(lambda x: x.toordinal()) 
                                            for col in joined_data3.select_dtypes(include=['datetime64'])})
        

        data_numeric = data_numeric.sort_values(by="T")

        print(data_numeric.head())

        # features = ["year", "month_num", "Season_Index", "precipitation", 
        #             "Forecasted Precipitation", "amountmilked", "months_gap"]
        features = ["year", "month_num", "Season_Index", "precipitation", 
                    "Forecasted Precipitation", "amountmilked", "months_gap","GrazingDist"]

        # Call the function Milk_forecasting funtion
        if data_numeric is None or data_numeric.empty:
            print("Warning: data_numeric is empty or None. Cannot proceed with forecasting.")

            results = {
                'model': None,
                'scaler': None,
                'training_history': None,
                'evaluation_metrics': {},
                'test_results': pd.DataFrame(),
                'forecast_results': pd.DataFrame(),
                'feature_indices': {},
                'data_month_to_season': {}
            }
            forecast_df = results["forecast_results"]
            test_results = results["test_results"]
            evaluation_metrics = results["evaluation_metrics"]

        else:

            results = GrazingDistForecaster(
                data_numeric.copy(),
                features=features,
                n_future=16,
                external_precip_forecasts=precipitation_forecasts_df#,
                # enhancement_method='enhanced_original'
            )
            # Access the results

            forecast_df = results["forecast_results"]
            test_results = results["test_results"]
            evaluation_metrics = results["evaluation_metrics"]

            print(results['forecast_results'][['Ward', 'Month', 'Year', 'Forecasted GrazingDist']].head())
            forecast_df.rename(columns={'Months Gap': 'Months_Gap','Forecasted GrazingDist': 'Forecasted_Value','Actual (if available)': 'Actual','Forecast Uncertainty (Std Dev)': 'Forecast_Uncertainty','Lower Bound (95%)': 'Lower_Bound','Upper Bound (95%)': 'Upper_Bound','Percent Error': 'Percent_Error'}, inplace=True)
            forecast_df5=forecast_df[['Month','Year','Season_Index','Precipitation','Months_Gap','Date','Date_Object','Forecasted_Value','Actual','Forecast_Uncertainty','Lower_Bound','Upper_Bound','Error','Percent_Error','Last_Actual_Value','Month1_Forecast','Month2_Forecast','Month3_Forecast','Evaluation_Metrics']]
            forecast_df5['Ward']=unique_ward2
            forecast_df5['Indicator']="DistInKmsToWaterSourceFromGrazingArea"
            forecast_df5=forecast_df5[['Ward','Month','Year','Season_Index','Precipitation','Months_Gap','Date','Date_Object','Indicator','Forecasted_Value','Actual','Forecast_Uncertainty','Lower_Bound','Upper_Bound','Error','Percent_Error','Last_Actual_Value','Month1_Forecast','Month2_Forecast','Month3_Forecast','Evaluation_Metrics']]
            forecast_df5

            # Inserting results into DB
            for col in forecast_df5.columns:

                if pd.api.types.is_numeric_dtype(forecast_df5[col]):
                    forecast_df5[col] = forecast_df5[col].apply(
                        lambda x: float(x) if not pd.isna(x) else None
                    )

                elif pd.api.types.is_datetime64_dtype(forecast_df5[col]):
                    forecast_df5[col] = forecast_df5[col].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(x) else None
                    )

                else:
                    forecast_df5[col] = forecast_df5[col].apply(
                        lambda x: str(x) if not pd.isna(x) else None
                    )

            if 'Date_Object' in forecast_df5.columns:
                forecast_df5['Date_Object'] = forecast_df5['Date_Object'].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') and x is not None 
                            else x if isinstance(x, str) else None
                )

            existing_special_rows = []
            try:
                query = text("""
                    SELECT Ward, Month, Year FROM Predictions 
                    WHERE Last_Actual_Value IS NOT NULL
                """)
                with engine.connect() as conn:
                    result = conn.execute(query)
                    existing_special_rows = [(row[0], row[1], row[2]) for row in result]
                print(f"Found {len(existing_special_rows)} existing special rows in database")
            except Exception as e:
                print(f"Error checking for existing special rows: {e}")

            special_rows_df = forecast_df5[forecast_df5['Last_Actual_Value'].notnull()].copy()
            special_rows_keys = [(row['Ward'], row['Month'], row['Year']) for _, row in special_rows_df.iterrows()]
            print(f"Found {len(special_rows_keys)} special rows in forecast_df5")

            insert_count = 0
            skip_count = 0

            for idx, row in forecast_df5.iterrows():
                ward = row['Ward']
                month = row['Month']
                year = row['Year']
                indicator = row['Indicator']
                row_key = (ward, month, year, indicator)
                is_special = row['Last_Actual_Value'] is not None

                if is_special and row_key not in existing_special_rows:
                    try:
                        delete_query = text("""
                            DELETE FROM Predictions 
                            WHERE Ward = :ward AND Month = :month AND Year = :year AND Indicator = :indicator
                        """)
                        with engine.begin() as conn:
                            conn.execute(delete_query, {"ward": ward, "month": month, "year": year, "indicator": indicator})
                        print(f"Cleared existing rows for new special row {row_key}")
                    except Exception as e:
                        print(f"Error clearing rows for new special row {row_key}: {e}")

                    should_insert = True

                elif not is_special and row_key not in existing_special_rows:

                    try:
                        delete_query = text("""
                            DELETE FROM Predictions 
                            WHERE Ward = :ward AND Month = :month AND Year = :year AND Indicator = :indicator
                        """)
                        with engine.begin() as conn:
                            conn.execute(delete_query, {"ward": ward, "month": month, "year": year, "indicator": indicator})
                    except Exception as e:
                        print(f"Error clearing rows for non-special row {row_key}: {e}")

                    should_insert = True
                
                elif is_special and row_key in existing_special_rows:

                    try:
                        delete_query = text("""
                            DELETE FROM Predictions 
                            WHERE Ward = :ward AND Month = :month AND Year = :year AND Indicator = :indicator
                        """)
                        with engine.begin() as conn:
                            conn.execute(delete_query, {"ward": ward, "month": month, "year": year, "indicator": indicator})
                        print(f"Deleted existing special row {row_key} for update")
                    except Exception as e:
                        print(f"Error deleting existing special row {row_key} for update: {e}")

                    should_insert = True

                else:  
                    print(f"Skipping non-special row {row_key} because special row exists in DB")
                    skip_count += 1
                    should_insert = False

                if should_insert:
                    params = {}
                    for col in forecast_df5.columns:
                        value = row[col]

                        if hasattr(value, 'dtype'):
                            if np.issubdtype(value.dtype, np.integer):
                                params[col.lower()] = int(value)
                            elif np.issubdtype(value.dtype, np.floating):
                                params[col.lower()] = float(value) if not np.isnan(value) else None
                            else:
                                params[col.lower()] = str(value) if value is not None else None
                        elif pd.isna(value):
                            params[col.lower()] = None
                        else:
                            params[col.lower()] = value

                    if 'indicator' not in params:
                        params['indicator'] = 'amountmilked'  
                                    
                    if 'forecasted_value' not in params and 'forecasted_amount_milked' in params:
                        params['forecasted_value'] = params['forecasted_amount_milked']
                                    
                    try:
                        inspect_query = text("""
                            SHOW COLUMNS FROM Predictions
                        """)
                        with engine.connect() as conn:
                            result = conn.execute(inspect_query)
                            db_columns = [row[0].lower() for row in result]
                            
                        
                        columns = []
                        values = []
                        insert_params = {}
                        
                        for key, value in params.items():
                            if key.lower() in db_columns:
                                columns.append(key.lower())
                                values.append(f":{key.lower()}")
                                insert_params[key.lower()] = value
                        
                        
                        columns_str = ", ".join(columns)
                        values_str = ", ".join(values)
                        insert_query = text(f"""
                            INSERT INTO Predictions ({columns_str}) 
                            VALUES ({values_str})
                        """)
                        
                        with engine.begin() as conn:
                            conn.execute(insert_query, insert_params)
                        
                        insert_count += 1
                        if is_special:
                            print(f"Inserted/updated special row {row_key}")
                        elif insert_count % 10 == 0:
                            print(f"Inserted {insert_count} rows so far")
                            
                    except Exception as e:
                        print(f"Error inserting record {row_key}: {e}")
                        print("Parameters used:")
                        for key, value in params.items():
                            print(f"  {key}: {value} (type: {type(value)})")

            print(f"Inserted/updated {insert_count} rows, skipped {skip_count} rows")
            print("Process completed successfully.")
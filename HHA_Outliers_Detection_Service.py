import pandas as pd
import numpy as np
from datetime import date
from sqlalchemy import create_engine, text
from Outliers_HHA import HHA_Outliers


# Function to convert NumPy types to Python native types
def convert_numpy_types(df):
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Convert all columns with numpy dtypes to Python native types
    for col in result.columns:
        if pd.api.types.is_integer_dtype(result[col]):
            result[col] = result[col].astype(int)
        elif pd.api.types.is_float_dtype(result[col]):
            result[col] = result[col].astype(float)
    
    return result

def process_outliers():

    # Create SQLAlchemy engine
    engine = create_engine(
        'mysql+mysqlconnector://root:Romans17:48@127.0.0.1/livelihoodzones_1'
    )


    hha_outliers = HHA_Outliers()
    #Milk_Prod_Cons_Sold
    query = """
        SELECT hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId, 
            hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId, hha_questionnaire_sessions.WardId,
            hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId, data_collection_exercise.DataCollectionExerciseId, 
            data_collection_exercise.ExerciseStartDate, hh_livestock_milk_production_per_species.AnimalId, hh_livestock_milk_production_per_species.DailyQntyMilkedInLtrs,
            hh_livestock_milk_production_per_species.DailyQntyConsumedInLtrs, hh_livestock_milk_production_per_species.DailyQntySoldInLtrs,
            hh_livestock_milk_production_per_species.PricePerLtr
        FROM (hh_livestock_milk_production_per_species
            LEFT JOIN hha_questionnaire_sessions ON (hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
            LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
        WHERE (hha_questionnaire_sessions.CountyId = '46' 
            AND data_collection_exercise.DataCollectionExerciseId = '106')
    """

    df_milk = pd.read_sql(query, engine)

    convert_numpy_types(df=df_milk)

    # Check if dataframe is empty before proceeding
    if df_milk.empty:
        print("Warning: No data returned from query")
        exit()

    #milk_outliers = Milk_Prod_Outliers()

    # Call the method with your dataframe
    results = hha_outliers.detect_outliers_multicolumn(df=df_milk)


    dataset_info = results['dataset_info']
    column_stats = results['column_stats']
    outliers_df = results['outliers']

    print("Dataset info:", dataset_info)
    print(f"Found {len(outliers_df)} outliers")

    # Check if we have any outliers before proceeding
    if outliers_df.empty:
        print("No outliers found. Exiting.")
        exit()

    outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
    outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
    outliers_df3 = outliers_df2.drop_duplicates()
    outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)
    outliers_df3['QuestionnaireType'] = "HHA"
    outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

    # Delete any previous outlier test instances
    outlier_run_names = outliers_df3['OutlierRunName'].tolist()
    data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


    for i in range(len(outlier_run_names)):
        del_query = text("""
            DELETE FROM outlier_runs 
            WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
        """)
        
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    del_query, 
                    {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                )
                print(f"Deleted {result.rowcount} previous outlier runs successfully")
        except Exception as e:
            print(f"Error occurred during deletion: {e}")

        # Insert new outlier test instances
    for _, row in outliers_df4.iterrows():
        insert_query = text("""
            INSERT INTO outlier_runs (
                OutlierRunName, DataCollectionExerciseId, QuestionnaireType
            ) 
            VALUES (:run_name, :exercise_id, :questionnaire_type)
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    insert_query, 
                    {
                        "run_name": row['OutlierRunName'],
                        "exercise_id": int(row['DataCollectionExerciseId']),
                        "questionnaire_type": row['QuestionnaireType']
                    }
                )
            print(f"Inserted outlier run record successfully")
        except Exception as e:
            print(f"Error occurred during outlier run insertion: {e}")

    # Get the OutlierRunId for the newly created run
    outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
    data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

    query = text("""
        SELECT outlier_runs.OutlierRunId
        FROM outlier_runs
        WHERE OutlierRunName = :run_name 
        AND DataCollectionExerciseId = :exercise_id
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query, 
            {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
        )
        OutlierRunId = result.scalar()

    if OutlierRunId is None:
        print("Error: Could not retrieve OutlierRunId. Exiting.")
        exit()

    print(f"Using OutlierRunId: {OutlierRunId}")

    # Add OutlierRunId to the outliers dataframe and convert numpy types
    outliers_df5 = outliers_df.copy()
    outliers_df5['OutlierRunId'] = OutlierRunId
    outliers_df5 = convert_numpy_types(outliers_df5)

    # Select and rename columns for the final insert
    outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                'IndicatorType', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                'test_type', 'level', 'reference_mean', 'reference_std', 
                                'population_mean', 'population_std']]

    outliers_df6.rename(columns={
        'analyzed_column': 'Indicator',
        'outlier_type': 'OutlierType',
        'test_statistic_value': 'TestStatisticValue',
        'test_type': 'TestType',
        'level': 'Level',
        'reference_mean': 'ReferenceMean',
        'reference_std': 'ReferenceStd',
        'population_mean': 'PopulationMean',
        'population_std': 'PopulationStd'
    }, inplace=True)

    # Insert outlier details
    for _, row in outliers_df6.iterrows():
        params = {}
        for col in outliers_df6.columns:
            value = row[col]
            if isinstance(value, float) and np.isnan(value):
                params[col.lower()] = None
            elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                params[col.lower()] = int(value)
            elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                params[col.lower()] = float(value)
            else:
                params[col.lower()] = value
        
        insert_query = text("""
            INSERT INTO outliers (
                OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                ReferenceMean, ReferenceStd, PopulationMean, PopulationStd
            ) 
            VALUES (
                :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                :referencemean, :referencestd, :populationmean, :populationstd
            )
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, params)
            
            # Print progress after every 100 rows
            if _ % 100 == 0:
                print(f"Inserted {_} outlier records...")
        except Exception as e:
            print(f"Error inserting record {_}: {e}")

    print(f"Inserted {len(outliers_df6)} outlier records successfully")


    #Copying_Strategies
    query = """
        SELECT hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId, 
            hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId, hha_questionnaire_sessions.WardId,
            hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId, data_collection_exercise.DataCollectionExerciseId, 
            data_collection_exercise.ExerciseStartDate, hh_livestock_milk_production_per_species.AnimalId, hh_livestock_milk_production_per_species.DailyQntyMilkedInLtrs,
            hh_livestock_milk_production_per_species.DailyQntyConsumedInLtrs, hh_livestock_milk_production_per_species.DailyQntySoldInLtrs,
            hh_livestock_milk_production_per_species.PricePerLtr
        FROM (hh_livestock_milk_production_per_species
            LEFT JOIN hha_questionnaire_sessions ON (hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
            LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
        WHERE (hha_questionnaire_sessions.CountyId = '46' 
            AND data_collection_exercise.DataCollectionExerciseId = '106')
    """

    coping_df = pd.read_sql(query, engine)

    convert_numpy_types(df=coping_df)

    # Check if dataframe is empty before proceeding
    if coping_df.empty:
        print("Warning: No data returned from query")
        exit()

        #copying_outliers = Copying_Strategies_Outliers()

        # Call the method with your dataframe
        results = hha_outliers.detect_outliers_Copying_Strategies(df=coping_df)


        dataset_info = results['dataset_info']
        column_stats = results['column_stats']
        outliers_df = results['outliers']

        print("Dataset info:", dataset_info)
        print(f"Found {len(outliers_df)} outliers")

    # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            exit()

        outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
        outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
        outliers_df3 = outliers_df2.drop_duplicates()
        outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)
        outliers_df3['QuestionnaireType'] = "HHA"
        outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

    # Delete any previous outlier test instances
        outlier_run_names = outliers_df3['OutlierRunName'].tolist()
        data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


        for i in range(len(outlier_run_names)):
            del_query = text("""
            DELETE FROM outlier_runs 
            WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
        """)
        
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    del_query, 
                    {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                )
                print(f"Deleted {result.rowcount} previous outlier runs successfully")
        except Exception as e:
            print(f"Error occurred during deletion: {e}")

        # Insert new outlier test instances
        for _, row in outliers_df4.iterrows():
            insert_query = text("""
            INSERT INTO outlier_runs (
                OutlierRunName, DataCollectionExerciseId, QuestionnaireType
            ) 
            VALUES (:run_name, :exercise_id, :questionnaire_type)
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    insert_query, 
                    {
                        "run_name": row['OutlierRunName'],
                        "exercise_id": int(row['DataCollectionExerciseId']),
                        "questionnaire_type": row['QuestionnaireType']
                    }
                )
            print(f"Inserted outlier run record successfully")
        except Exception as e:
            print(f"Error occurred during outlier run insertion: {e}")

    # Get the OutlierRunId for the newly created run
        outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
        data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

        query = text("""
        SELECT outlier_runs.OutlierRunId
        FROM outlier_runs
        WHERE OutlierRunName = :run_name 
        AND DataCollectionExerciseId = :exercise_id
    """)

        with engine.connect() as conn:
            result = conn.execute(
                query, 
                {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
            )
            OutlierRunId = result.scalar()

        if OutlierRunId is None:
            print("Error: Could not retrieve OutlierRunId. Exiting.")
            exit()

        print(f"Using OutlierRunId: {OutlierRunId}")

        # Add OutlierRunId to the outliers dataframe and convert numpy types
        outliers_df5 = outliers_df.copy()
        outliers_df5['OutlierRunId'] = OutlierRunId
        outliers_df5 = convert_numpy_types(outliers_df5)

    # Select and rename columns for the final insert
        outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                'IndicatorType', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                'test_type', 'level', 'reference_mean', 'reference_std', 
                                'population_mean', 'population_std']]

        outliers_df6.rename(columns={
        'analyzed_column': 'Indicator',
        'outlier_type': 'OutlierType',
        'test_statistic_value': 'TestStatisticValue',
        'test_type': 'TestType',
        'level': 'Level',
        'reference_mean': 'ReferenceMean',
        'reference_std': 'ReferenceStd',
        'population_mean': 'PopulationMean',
        'population_std': 'PopulationStd'
        }, inplace=True)

    # Insert outlier details
        for _, row in outliers_df6.iterrows():
            params = {}
            for col in outliers_df6.columns:
                value = row[col]
                if isinstance(value, float) and np.isnan(value):
                    params[col.lower()] = None
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                    params[col.lower()] = int(value)
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                    params[col.lower()] = float(value)
                else:
                    params[col.lower()] = value
        
            insert_query = text("""
                INSERT INTO outliers (
                OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                ReferenceMean, ReferenceStd, PopulationMean, PopulationStd
                ) 
                VALUES (
                :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                :referencemean, :referencestd, :populationmean, :populationstd
            )
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, params)
            
            # Print progress after every 100 rows
            if _ % 100 == 0:
                print(f"Inserted {_} outlier records...")
        except Exception as e:
            print(f"Error inserting record {_}: {e}")

        print(f"Inserted {len(outliers_df6)} outlier records successfully")


    #Crop_production
    query = """
            SELECT 
                hcp.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
                dce.DataCollectionExerciseId, dce.ExerciseStartDate, hcp.CropId,
                hcp.AcresPlantedInLastFourWks, hcp.AcresHarvestedInLastFourWks,
                hcp.KgsHarvestedInLastFourWks, hcp.OwnProductionStockInKg,
                hcp.KgsSoldInLastFourWks, hcp.PricePerKg
            FROM hh_crop_production_per_species AS hcp
            LEFT JOIN hha_questionnaire_sessions AS hhs ON hcp.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
            LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
            WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
        """

    crop_df = pd.read_sql(query, engine)

    convert_numpy_types(df=crop_df)

    # Check if dataframe is empty before proceeding
    if coping_df.empty:
        print("Warning: No data returned from query")
        exit()

        #copying_outliers = Crop_Outliers()

        # Call the method with your dataframe
        results = hha_outliers.detect_outliers_crop_production(df=crop_df)


        dataset_info = results['dataset_info']
        column_stats = results['column_stats']
        outliers_df = results['outliers']

        print("Dataset info:", dataset_info)
        print(f"Found {len(outliers_df)} outliers")

    # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            exit()

        outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
        outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
        outliers_df3 = outliers_df2.drop_duplicates()
        outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)
        outliers_df3['QuestionnaireType'] = "HHA"
        outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

    # Delete any previous outlier test instances
        outlier_run_names = outliers_df3['OutlierRunName'].tolist()
        data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


        for i in range(len(outlier_run_names)):
            del_query = text("""
            DELETE FROM outlier_runs 
            WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
        """)
        
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    del_query, 
                    {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                )
                print(f"Deleted {result.rowcount} previous outlier runs successfully")
        except Exception as e:
            print(f"Error occurred during deletion: {e}")

        # Insert new outlier test instances
        for _, row in outliers_df4.iterrows():
            insert_query = text("""
            INSERT INTO outlier_runs (
                OutlierRunName, DataCollectionExerciseId, QuestionnaireType
            ) 
            VALUES (:run_name, :exercise_id, :questionnaire_type)
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    insert_query, 
                    {
                        "run_name": row['OutlierRunName'],
                        "exercise_id": int(row['DataCollectionExerciseId']),
                        "questionnaire_type": row['QuestionnaireType']
                    }
                )
            print(f"Inserted outlier run record successfully")
        except Exception as e:
            print(f"Error occurred during outlier run insertion: {e}")

    # Get the OutlierRunId for the newly created run
        outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
        data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

        query = text("""
        SELECT outlier_runs.OutlierRunId
        FROM outlier_runs
        WHERE OutlierRunName = :run_name 
        AND DataCollectionExerciseId = :exercise_id
    """)

        with engine.connect() as conn:
            result = conn.execute(
                query, 
                {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
            )
            OutlierRunId = result.scalar()

        if OutlierRunId is None:
            print("Error: Could not retrieve OutlierRunId. Exiting.")
            exit()

        print(f"Using OutlierRunId: {OutlierRunId}")

        # Add OutlierRunId to the outliers dataframe and convert numpy types
        outliers_df5 = outliers_df.copy()
        outliers_df5['OutlierRunId'] = OutlierRunId
        outliers_df5 = convert_numpy_types(outliers_df5)

    # Select and rename columns for the final insert
        outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                'IndicatorType', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                'test_type', 'level', 'reference_mean', 'reference_std', 
                                'population_mean', 'population_std']]

        outliers_df6.rename(columns={
        'analyzed_column': 'Indicator',
        'outlier_type': 'OutlierType',
        'test_statistic_value': 'TestStatisticValue',
        'test_type': 'TestType',
        'level': 'Level',
        'reference_mean': 'ReferenceMean',
        'reference_std': 'ReferenceStd',
        'population_mean': 'PopulationMean',
        'population_std': 'PopulationStd'
        }, inplace=True)

    # Insert outlier details
        for _, row in outliers_df6.iterrows():
            params = {}
            for col in outliers_df6.columns:
                value = row[col]
                if isinstance(value, float) and np.isnan(value):
                    params[col.lower()] = None
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                    params[col.lower()] = int(value)
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                    params[col.lower()] = float(value)
                else:
                    params[col.lower()] = value
        
            insert_query = text("""
                INSERT INTO outliers (
                OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                ReferenceMean, ReferenceStd, PopulationMean, PopulationStd
                ) 
                VALUES (
                :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                :referencemean, :referencestd, :populationmean, :populationstd
            )
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, params)
            
            # Print progress after every 100 rows
            if _ % 100 == 0:
                print(f"Inserted {_} outlier records...")
        except Exception as e:
            print(f"Error inserting record {_}: {e}")

        print(f"Inserted {len(outliers_df6)} outlier records successfully")


    #Crop_production
    query = """
            SELECT 
                hcp.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
                dce.DataCollectionExerciseId, dce.ExerciseStartDate, hcp.CropId,
                hcp.AcresPlantedInLastFourWks, hcp.AcresHarvestedInLastFourWks,
                hcp.KgsHarvestedInLastFourWks, hcp.OwnProductionStockInKg,
                hcp.KgsSoldInLastFourWks, hcp.PricePerKg
            FROM hh_crop_production_per_species AS hcp
            LEFT JOIN hha_questionnaire_sessions AS hhs ON hcp.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
            LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
            WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
        """

    crop_df = pd.read_sql(query, engine)

    convert_numpy_types(df=crop_df)

    # Check if dataframe is empty before proceeding
    if coping_df.empty:
        print("Warning: No data returned from query")
        exit()

        #copying_outliers = Crop_Outliers()

        # Call the method with your dataframe
        results = hha_outliers.detect_outliers_crop_production(df=crop_df)


        dataset_info = results['dataset_info']
        column_stats = results['column_stats']
        outliers_df = results['outliers']

        print("Dataset info:", dataset_info)
        print(f"Found {len(outliers_df)} outliers")

    # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            exit()

        outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
        outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
        outliers_df3 = outliers_df2.drop_duplicates()
        outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)
        outliers_df3['QuestionnaireType'] = "HHA"
        outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

    # Delete any previous outlier test instances
        outlier_run_names = outliers_df3['OutlierRunName'].tolist()
        data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


        for i in range(len(outlier_run_names)):
            del_query = text("""
            DELETE FROM outlier_runs 
            WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
        """)
        
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    del_query, 
                    {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                )
                print(f"Deleted {result.rowcount} previous outlier runs successfully")
        except Exception as e:
            print(f"Error occurred during deletion: {e}")

        # Insert new outlier test instances
        for _, row in outliers_df4.iterrows():
            insert_query = text("""
            INSERT INTO outlier_runs (
                OutlierRunName, DataCollectionExerciseId, QuestionnaireType
            ) 
            VALUES (:run_name, :exercise_id, :questionnaire_type)
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    insert_query, 
                    {
                        "run_name": row['OutlierRunName'],
                        "exercise_id": int(row['DataCollectionExerciseId']),
                        "questionnaire_type": row['QuestionnaireType']
                    }
                )
            print(f"Inserted outlier run record successfully")
        except Exception as e:
            print(f"Error occurred during outlier run insertion: {e}")

    # Get the OutlierRunId for the newly created run
        outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
        data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

        query = text("""
        SELECT outlier_runs.OutlierRunId
        FROM outlier_runs
        WHERE OutlierRunName = :run_name 
        AND DataCollectionExerciseId = :exercise_id
    """)

        with engine.connect() as conn:
            result = conn.execute(
                query, 
                {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
            )
            OutlierRunId = result.scalar()

        if OutlierRunId is None:
            print("Error: Could not retrieve OutlierRunId. Exiting.")
            exit()

        print(f"Using OutlierRunId: {OutlierRunId}")

        # Add OutlierRunId to the outliers dataframe and convert numpy types
        outliers_df5 = outliers_df.copy()
        outliers_df5['OutlierRunId'] = OutlierRunId
        outliers_df5 = convert_numpy_types(outliers_df5)

    # Select and rename columns for the final insert
        outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                'IndicatorType', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                'test_type', 'level', 'reference_mean', 'reference_std', 
                                'population_mean', 'population_std']]

        outliers_df6.rename(columns={
        'analyzed_column': 'Indicator',
        'outlier_type': 'OutlierType',
        'test_statistic_value': 'TestStatisticValue',
        'test_type': 'TestType',
        'level': 'Level',
        'reference_mean': 'ReferenceMean',
        'reference_std': 'ReferenceStd',
        'population_mean': 'PopulationMean',
        'population_std': 'PopulationStd'
        }, inplace=True)

    # Insert outlier details
        for _, row in outliers_df6.iterrows():
            params = {}
            for col in outliers_df6.columns:
                value = row[col]
                if isinstance(value, float) and np.isnan(value):
                    params[col.lower()] = None
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                    params[col.lower()] = int(value)
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                    params[col.lower()] = float(value)
                else:
                    params[col.lower()] = value
        
            insert_query = text("""
                INSERT INTO outliers (
                OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                ReferenceMean, ReferenceStd, PopulationMean, PopulationStd
                ) 
                VALUES (
                :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                :referencemean, :referencestd, :populationmean, :populationstd
            )
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, params)
            
            # Print progress after every 100 rows
            if _ % 100 == 0:
                print(f"Inserted {_} outlier records...")
        except Exception as e:
            print(f"Error inserting record {_}: {e}")

        print(f"Inserted {len(outliers_df6)} outlier records successfully")



    #Livestock_production
    query = """
            SELECT 
                hlp.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
                dce.DataCollectionExerciseId, dce.ExerciseStartDate, hlp.AnimalId,
                hlp.NumberKeptToday, hlp.NumberBornInLastFourWeeks,
                hlp.NumberPurchasedInLastFourWeeks, hlp.NumberSoldInLastFourWeeks,
                hlp.AveragePricePerAnimalSold, hlp.NumberDiedDuringLastFourWeeks
            FROM hh_livestock_production_by_species AS hlp
            LEFT JOIN hha_questionnaire_sessions AS hhs ON hlp.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
            LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
            WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
        """

    livestock_df = pd.read_sql(query, engine)

    convert_numpy_types(df=livestock_df)

    # Check if dataframe is empty before proceeding
    if coping_df.empty:
        print("Warning: No data returned from query")
        exit()

        #copying_outliers = Crop_Outliers()

        # Call the method with your dataframe
        results = hha_outliers.detect_outliers_livestock_production(df=livestock_df)


        dataset_info = results['dataset_info']
        column_stats = results['column_stats']
        outliers_df = results['outliers']

        print("Dataset info:", dataset_info)
        print(f"Found {len(outliers_df)} outliers")

    # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            exit()

        outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
        outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
        outliers_df3 = outliers_df2.drop_duplicates()
        outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)
        outliers_df3['QuestionnaireType'] = "HHA"
        outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

    # Delete any previous outlier test instances
        outlier_run_names = outliers_df3['OutlierRunName'].tolist()
        data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


        for i in range(len(outlier_run_names)):
            del_query = text("""
            DELETE FROM outlier_runs 
            WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
        """)
        
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    del_query, 
                    {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                )
                print(f"Deleted {result.rowcount} previous outlier runs successfully")
        except Exception as e:
            print(f"Error occurred during deletion: {e}")

        # Insert new outlier test instances
        for _, row in outliers_df4.iterrows():
            insert_query = text("""
            INSERT INTO outlier_runs (
                OutlierRunName, DataCollectionExerciseId, QuestionnaireType
            ) 
            VALUES (:run_name, :exercise_id, :questionnaire_type)
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    insert_query, 
                    {
                        "run_name": row['OutlierRunName'],
                        "exercise_id": int(row['DataCollectionExerciseId']),
                        "questionnaire_type": row['QuestionnaireType']
                    }
                )
            print(f"Inserted outlier run record successfully")
        except Exception as e:
            print(f"Error occurred during outlier run insertion: {e}")

    # Get the OutlierRunId for the newly created run
        outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
        data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

        query = text("""
        SELECT outlier_runs.OutlierRunId
        FROM outlier_runs
        WHERE OutlierRunName = :run_name 
        AND DataCollectionExerciseId = :exercise_id
    """)

        with engine.connect() as conn:
            result = conn.execute(
                query, 
                {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
            )
            OutlierRunId = result.scalar()

        if OutlierRunId is None:
            print("Error: Could not retrieve OutlierRunId. Exiting.")
            exit()

        print(f"Using OutlierRunId: {OutlierRunId}")

        # Add OutlierRunId to the outliers dataframe and convert numpy types
        outliers_df5 = outliers_df.copy()
        outliers_df5['OutlierRunId'] = OutlierRunId
        outliers_df5 = convert_numpy_types(outliers_df5)

    # Select and rename columns for the final insert
        outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                'IndicatorType', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                'test_type', 'level', 'reference_mean', 'reference_std', 
                                'population_mean', 'population_std']]

        outliers_df6.rename(columns={
        'analyzed_column': 'Indicator',
        'outlier_type': 'OutlierType',
        'test_statistic_value': 'TestStatisticValue',
        'test_type': 'TestType',
        'level': 'Level',
        'reference_mean': 'ReferenceMean',
        'reference_std': 'ReferenceStd',
        'population_mean': 'PopulationMean',
        'population_std': 'PopulationStd'
        }, inplace=True)

    # Insert outlier details
        for _, row in outliers_df6.iterrows():
            params = {}
            for col in outliers_df6.columns:
                value = row[col]
                if isinstance(value, float) and np.isnan(value):
                    params[col.lower()] = None
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                    params[col.lower()] = int(value)
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                    params[col.lower()] = float(value)
                else:
                    params[col.lower()] = value
        
            insert_query = text("""
                INSERT INTO outliers (
                OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                ReferenceMean, ReferenceStd, PopulationMean, PopulationStd
                ) 
                VALUES (
                :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                :referencemean, :referencestd, :populationmean, :populationstd
            )
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, params)
            
            # Print progress after every 100 rows
            if _ % 100 == 0:
                print(f"Inserted {_} outlier records...")
        except Exception as e:
            print(f"Error inserting record {_}: {e}")

        print(f"Inserted {len(outliers_df6)} outlier records successfully")



    #Livestock_production
    query = """
            SELECT 
                hfc.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
                dce.DataCollectionExerciseId, dce.ExerciseStartDate, hfc.FoodTypeId,
                hfc.NumDaysEaten
            FROM hh_food_consumption AS hfc
            LEFT JOIN hha_questionnaire_sessions AS hhs ON hfc.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
            LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
            WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
        """

    food_c_df = pd.read_sql(query, engine)
    convert_numpy_types(df=food_c_df)

    # Check if dataframe is empty before proceeding
    if coping_df.empty:
        print("Warning: No data returned from query")
        exit()

        #copying_outliers = Crop_Outliers()

        # Call the method with your dataframe
        results = hha_outliers.detect_outliers_Food_Consumption(df=food_c_df)


        dataset_info = results['dataset_info']
        column_stats = results['column_stats']
        outliers_df = results['outliers']

        print("Dataset info:", dataset_info)
        print(f"Found {len(outliers_df)} outliers")

    # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            exit()

        outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
        outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
        outliers_df3 = outliers_df2.drop_duplicates()
        outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)
        outliers_df3['QuestionnaireType'] = "HHA"
        outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

    # Delete any previous outlier test instances
        outlier_run_names = outliers_df3['OutlierRunName'].tolist()
        data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


        for i in range(len(outlier_run_names)):
            del_query = text("""
            DELETE FROM outlier_runs 
            WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
        """)
        
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    del_query, 
                    {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                )
                print(f"Deleted {result.rowcount} previous outlier runs successfully")
        except Exception as e:
            print(f"Error occurred during deletion: {e}")

        # Insert new outlier test instances
        for _, row in outliers_df4.iterrows():
            insert_query = text("""
            INSERT INTO outlier_runs (
                OutlierRunName, DataCollectionExerciseId, QuestionnaireType
            ) 
            VALUES (:run_name, :exercise_id, :questionnaire_type)
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(
                    insert_query, 
                    {
                        "run_name": row['OutlierRunName'],
                        "exercise_id": int(row['DataCollectionExerciseId']),
                        "questionnaire_type": row['QuestionnaireType']
                    }
                )
            print(f"Inserted outlier run record successfully")
        except Exception as e:
            print(f"Error occurred during outlier run insertion: {e}")

    # Get the OutlierRunId for the newly created run
        outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
        data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

        query = text("""
        SELECT outlier_runs.OutlierRunId
        FROM outlier_runs
        WHERE OutlierRunName = :run_name 
        AND DataCollectionExerciseId = :exercise_id
    """)

        with engine.connect() as conn:
            result = conn.execute(
                query, 
                {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
            )
            OutlierRunId = result.scalar()

        if OutlierRunId is None:
            print("Error: Could not retrieve OutlierRunId. Exiting.")
            exit()

        print(f"Using OutlierRunId: {OutlierRunId}")

        # Add OutlierRunId to the outliers dataframe and convert numpy types
        outliers_df5 = outliers_df.copy()
        outliers_df5['OutlierRunId'] = OutlierRunId
        outliers_df5 = convert_numpy_types(outliers_df5)

    # Select and rename columns for the final insert
        outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                'IndicatorType', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                'test_type', 'level', 'reference_mean', 'reference_std', 
                                'population_mean', 'population_std']]

        outliers_df6.rename(columns={
        'analyzed_column': 'Indicator',
        'outlier_type': 'OutlierType',
        'test_statistic_value': 'TestStatisticValue',
        'test_type': 'TestType',
        'level': 'Level',
        'reference_mean': 'ReferenceMean',
        'reference_std': 'ReferenceStd',
        'population_mean': 'PopulationMean',
        'population_std': 'PopulationStd'
        }, inplace=True)

    # Insert outlier details
        for _, row in outliers_df6.iterrows():
            params = {}
            for col in outliers_df6.columns:
                value = row[col]
                if isinstance(value, float) and np.isnan(value):
                    params[col.lower()] = None
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                    params[col.lower()] = int(value)
                elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                    params[col.lower()] = float(value)
                else:
                    params[col.lower()] = value
        
            insert_query = text("""
                INSERT INTO outliers (
                OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                ReferenceMean, ReferenceStd, PopulationMean, PopulationStd
                ) 
                VALUES (
                :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                :referencemean, :referencestd, :populationmean, :populationstd
            )
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, params)
            
            # Print progress after every 100 rows
            if _ % 100 == 0:
                print(f"Inserted {_} outlier records...")
        except Exception as e:
            print(f"Error inserting record {_}: {e}")

        print(f"Inserted {len(outliers_df6)} outlier records successfully")


    print("Process completed successfully.")
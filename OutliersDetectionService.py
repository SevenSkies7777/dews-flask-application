import pandas as pd
from sqlalchemy import create_engine
import json
from HHA_Outliers_Test import (
    detect_outliers_crop_production,
    detect_outliers_livestock_production,
    detect_outliers_milk,
    detect_outliers_Copying_Strategies,
    detect_outliers_Food_Consumption
)

def fetch_outliers():
    """Fetches outliers from different datasets and returns them as JSON."""

    # Establish database connection
    engine = create_engine("mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones")

    # Load datasets with detailed queries
    crop_query = """
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

    livestock_query = """
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

    milk_query = """
        SELECT 
            hlm.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
            dce.DataCollectionExerciseId, dce.ExerciseStartDate, hlm.AnimalId,
            hlm.DailyQntyMilkedInLtrs, hlm.DailyQntyConsumedInLtrs,
            hlm.DailyQntySoldInLtrs, hlm.PricePerLtr
        FROM hh_livestock_milk_production_per_species AS hlm
        LEFT JOIN hha_questionnaire_sessions AS hhs ON hlm.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
    """

    coping_query = """
        SELECT 
            hcc.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
            dce.DataCollectionExerciseId, dce.ExerciseStartDate, hcc.CopyingStrategyId,
            hcc.NumOfCopingDays
        FROM hh_consumption_coping_strategies AS hcc
        LEFT JOIN hha_questionnaire_sessions AS hhs ON hcc.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
    """

    food_query = """
        SELECT 
            hfc.HhaQuestionnaireSessionId, hhs.CountyId, hhs.LivelihoodZoneId, hhs.WardId, hhs.HouseHoldId, hhs.SubCountyId,
            dce.DataCollectionExerciseId, dce.ExerciseStartDate, hfc.FoodTypeId,
            hfc.NumDaysEaten
        FROM hh_food_consumption AS hfc
        LEFT JOIN hha_questionnaire_sessions AS hhs ON hfc.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = '32' AND dce.DataCollectionExerciseId = '4';
    """

    # Read data from MySQL into Pandas DataFrames
    crop_df = pd.read_sql(crop_query, engine)
    livestock_df = pd.read_sql(livestock_query, engine)
    milk_df = pd.read_sql(milk_query, engine)
    coping_df = pd.read_sql(coping_query, engine)
    food_c_df = pd.read_sql(food_query, engine)

    # Handle missing values (convert NaN to appropriate values)
    for df in [crop_df, livestock_df, milk_df, coping_df, food_c_df]:
        df.fillna("", inplace=True)  # Replace NaN values with empty strings to prevent errors in JSON serialization

    # Perform outlier detection
    crop_outliers = detect_outliers_crop_production(crop_df)
    livestock_outliers = detect_outliers_livestock_production(livestock_df)
    milk_outliers = detect_outliers_milk(milk_df)
    coping_outliers = detect_outliers_Copying_Strategies(coping_df)
    food_outliers = detect_outliers_Food_Consumption(food_c_df)

    # Organize the results
    results = {
      "crop_production_outliers": crop_outliers,
      "livestock_production_outliers": livestock_outliers,
      "milk_production_outliers": milk_outliers,
      "coping_strategies_outliers": coping_outliers,
      "food_consumption_outliers": food_outliers
    }

    # ✅ Return a Python dictionary (not a JSON string)
    return results

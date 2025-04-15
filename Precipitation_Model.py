

#LSTM MOdel Precipitation
import numpy as np
import mysql.connector
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Forecast_utils import run_precip_forecast_pipeline

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
    
    results = run_precip_forecast_pipeline(prep_df)
    model = results["model"]
    forecast_df = results["forecast_df"]
    metrics = results["metrics"]

    print(forecast_df.head())
    print(metrics)







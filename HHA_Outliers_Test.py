#importing libraries
import pandas as pd
import numpy as np
import mysql.connector
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Romans17:48',
        database='livelihoodzones2'
    )

#Crop production

query = """
    SELECT hh_crop_production_per_species.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_crop_production_per_species.CropId,hh_crop_production_per_species.AcresPlantedInLastFourWks,hh_crop_production_per_species.AcresHarvestedInLastFourWks,hh_crop_production_per_species.KgsHarvestedInLastFourWks,hh_crop_production_per_species.OwnProductionStockInKg,hh_crop_production_per_species.KgsSoldInLastFourWks,hh_crop_production_per_species.PricePerKg
    FROM (hh_crop_production_per_species
          LEFT JOIN hha_questionnaire_sessions ON (hh_crop_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""

crop_df = pd.read_sql(query, conn)

#conn.close()

# Create a date field from month and year
#crop_df['date'] = pd.to_datetime(df['month'].astype(str) + ' ' + df['year'].astype(str), format='%B %Y')

#crop_df


#Crop production

def detect_outliers_crop_production(crop_df, id_column='CropId', 
                               value_columns=['AcresPlantedInLastFourWks', 'AcresHarvestedInLastFourWks', 
                                             'KgsHarvestedInLastFourWks', 'OwnProductionStockInKg', 'KgsSoldInLastFourWks', 'PricePerKg'],
                               z_threshold=2.5):
    """
    Comprehensive outlier detection for crop production data with multiple measurements
    Handles both univariate and multivariate outliers across multiple value columns
    Adapts to data size and checks for zero variance
    
    Parameters:
    -----------
    crop_df : pandas DataFrame
        Input data with columns WardId, CountyId, CropId, and multiple measurement columns
    id_column : str
        Name of the crop ID column
    value_columns : list
        List of measurement column names to analyze
    z_threshold : float
        Z-score threshold for univariate outlier detection
    """
    results = {
        'dataset_info': {},
        'column_stats': {},
        'univariate': {col: {} for col in value_columns},
        'multivariate': {},
        'ward_level': {}
    }
    print("\nCROP PRODUCTION OUTLIER ANALYSIS")
    print("=" * 50)    
    print("Starting outlier analysis...")

    
    # 1. Dataset Overview
    results['dataset_info'] = {
        'total_records': len(crop_df),
        'unique_wards': crop_df['WardId'].nunique(),
        'unique_crops': crop_df[id_column].nunique(),
        'unique_households': crop_df['HouseHoldId'].nunique() if 'HouseHoldId' in crop_df.columns else 'N/A'
    }
    
    print("\nDATASET INFORMATION")
    print("=" * 50)
    for key, value in results['dataset_info'].items():
        print(f"{key}: {value}")
    
    # 2. Check stats for each column
    print("\nCOLUMN STATISTICS")
    print("=" * 50)
    
    for column in value_columns:
        # Skip if column doesn't exist in dataframe
        if column not in crop_df.columns:
            print(f"Column {column} not found in dataframe. Skipping.")
            continue
            
        column_stats = {
            'mean': crop_df[column].mean(),
            'median': crop_df[column].median(),
            'std': crop_df[column].std(),
            'min': crop_df[column].min(),
            'max': crop_df[column].max(),
            'null_count': crop_df[column].isnull().sum(),
            'zero_count': (crop_df[column] == 0).sum()
        }
        results['column_stats'][column] = column_stats
        
        print(f"\nColumn: {column}")
        for stat_name, stat_value in column_stats.items():
            print(f"{stat_name}: {stat_value}")
    
    # 3. Process each measurement column
    for value_column in value_columns:
        if value_column not in crop_df.columns:
            continue
        
        print(f"\n\nANALYZING COLUMN: {value_column}")
        print("=" * 50)
        
        # 3.1 Check for zero variance crops in this column
        print("\nCHECKING VARIANCE")
        print("-" * 40)
        
        zero_var_crops = []
        valid_crops = []
        
        for crop in crop_df[id_column].unique():
            crop_data = crop_df[crop_df[id_column] == crop][value_column]
            # Skip if all values are null
            if crop_data.isnull().all():
                print(f"Crop {crop}: All values are NULL for {value_column}")
                continue
                
            # Handle nulls by dropping them for variance check
            crop_data = crop_data.dropna()
            
            if len(crop_data) == 0 or crop_data.std() == 0:
                zero_var_crops.append(crop)
                if len(crop_data) > 0:
                    constant_value = crop_data.iloc[0]
                    print(f"Crop {crop}: Zero variance (all values = {constant_value})")
                else:
                    print(f"Crop {crop}: No valid data after removing nulls")
            else:
                valid_crops.append(crop)
                print(f"Crop {crop}: Has variation")
        
        # 3.2 Univariate Outlier Detection for this column
        print("\nCOUNTY-LEVEL ANALYSIS")
        print("=" * 50)
        print(f"\nUNIVARIATE OUTLIER DETECTION FOR {value_column}")
        print("-" * 40)
        
        for crop in valid_crops:
            crop_data = crop_df[crop_df[id_column] == crop].copy()
            # Remove nulls for z-score calculation
            crop_data = crop_data.dropna(subset=[value_column])
            
            if len(crop_data) <= 1:
                print(f"Crop {crop}: Insufficient data for z-score calculation")
                continue
                
            crop_data['z_score'] = np.abs(stats.zscore(crop_data[value_column]))
            outliers = crop_data[crop_data['z_score'] > z_threshold]
            
            results['univariate'][value_column][crop] = {
                'total_observations': len(crop_data),
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers)/len(crop_data)*100) if len(crop_data) > 0 else 0,
                'mean': crop_data[value_column].mean(),
                'std': crop_data[value_column].std(),
                'outliers': outliers
            }
            
            print(f"\nCrop {crop}:")
            print(f"Total valid observations: {len(crop_data)}")
            print(f"Outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                print(f"Outlier percentage: {(len(outliers)/len(crop_data)*100):.2f}%")
                if 'WardId' in outliers.columns and 'HouseHoldId' in outliers.columns:
                    outlier_info = outliers[[value_column, 'z_score', 'WardId', 'HouseHoldId']]
                else:
                    outlier_info = outliers[[value_column, 'z_score']]
                print("\nOutlier details:")
                print(outlier_info.head(5))  # Show first 5 outliers only
                if len(outliers) > 5:
                    print(f"...and {len(outliers)-5} more outliers")
    
    # 4. Multivariate Outlier Detection across all value columns
    print("\nMULTIVARIATE OUTLIER DETECTION")
    print("=" * 50)
    
    # Filter to columns that exist in the dataframe
    available_columns = [col for col in value_columns if col in crop_df.columns]
    
    if len(available_columns) >= 2:  # Need at least 2 variables for multivariate analysis
        try:
            # Create a copy for multivariate analysis
            mv_crop_df = crop_df.copy()
            
            # For multivariate analysis, we need to handle missing values
            # Strategy: Impute with median for each crop
            for crop in crop_df[id_column].unique():
                crop_mask = mv_crop_df[id_column] == crop
                for col in available_columns:
                    median_val = mv_crop_df.loc[crop_mask, col].median()
                    # Only fill if median is not NaN
                    if pd.notna(median_val):
                        mv_crop_df.loc[crop_mask, col] = mv_crop_df.loc[crop_mask, col].fillna(median_val)
            
            # Drop any remaining rows with NaN after imputation
            mv_crop_df = mv_crop_df.dropna(subset=available_columns)
            
            if len(mv_crop_df) < 10:  # Not enough data for meaningful multivariate analysis
                print("Insufficient data for multivariate analysis after handling missing values")
                results['multivariate'] = {'error': 'Insufficient data after handling missing values'}
            else:
                # Group by IDs for household-level analysis
                id_columns = ['HouseHoldId', 'WardId', 'CountyId'] if 'HouseHoldId' in mv_crop_df.columns else ['WardId', 'CountyId']
                available_id_columns = [col for col in id_columns if col in mv_crop_df.columns]
                
                if not available_id_columns:
                    print("No ID columns available for grouping")
                    results['multivariate'] = {'error': 'No ID columns available for grouping'}
                else:
                    # Calculate Mahalanobis distance
                    X = mv_crop_df[available_columns].values
                    mean_vec = np.mean(X, axis=0)
                    cov_matrix = np.cov(X, rowvar=False)
                    
                    # Handle potential singularity with pseudo-inverse
                    try:
                        inv_covmat = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
                        inv_covmat = np.linalg.pinv(cov_matrix)
                    
                    mv_crop_df['mahalanobis_dist'] = np.zeros(len(mv_crop_df))
                    for idx in range(len(X)):
                        diff = X[idx] - mean_vec
                        mv_crop_df.loc[mv_crop_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))
                    
                    # Chi-square threshold for multivariate outliers
                    threshold = chi2.ppf(0.975, crop_df=len(available_columns))
                    mv_crop_df['multivar_outlier'] = mv_crop_df['mahalanobis_dist'] > threshold
                    
                    multivar_outliers = mv_crop_df[mv_crop_df['multivar_outlier']]
                    
                    results['multivariate'] = {
                        'total_observations': len(mv_crop_df),
                        'outliers_count': len(multivar_outliers),
                        'outliers_percentage': (len(multivar_outliers)/len(mv_crop_df)*100),
                        'mahalanobis_threshold': threshold,
                        'variables_used': available_columns,
                        'outliers': multivar_outliers
                    }
                    
                    print(f"\nMultivariate analysis results:")
                    print(f"Total observations: {len(mv_crop_df)}")
                    print(f"Variables used: {available_columns}")
                    print(f"Outliers detected: {len(multivar_outliers)} ({(len(multivar_outliers)/len(mv_crop_df)*100):.2f}%)")
                    if len(multivar_outliers) > 0:
                        display_columns = available_id_columns + ['mahalanobis_dist'] + available_columns
                        print("\nMultivariate outlier details (first 5):")
                        print(multivar_outliers[display_columns].head(5))
                        if len(multivar_outliers) > 5:
                            print(f"...and {len(multivar_outliers)-5} more outliers")
        
        except Exception as e:
            print(f"Note: Multivariate analysis could not be completed: {str(e)}")
            results['multivariate'] = {'error': str(e)}
    else:
        print(f"Insufficient variables for multivariate analysis. Need at least 2, found {len(available_columns)}")
        results['multivariate'] = {'error': 'Insufficient variables with variation'}
    
    # 5. Ward-level Analysis
    if 'WardId' in crop_df.columns:
        print("\nWARD-LEVEL ANALYSIS")
        print("=" * 50)
        
        for ward in crop_df['WardId'].unique():
            ward_data = crop_df[crop_df['WardId'] == ward]
            results['ward_level'][ward] = {
                'univariate': {col: {} for col in value_columns},
            }
            
            print(f"\nWard {ward}:")
            
            # Ward-level univariate outliers for each column
            for value_column in value_columns:
                if value_column not in crop_df.columns:
                    continue
                    
                for crop in crop_df[id_column].unique():
                    crop_ward_data = ward_data[ward_data[id_column] == crop].copy()
                    
                    # Skip if not enough data
                    if len(crop_ward_data) <= 1 or crop_ward_data[value_column].isnull().all():
                        continue
                        
                    # Remove nulls
                    crop_ward_data = crop_ward_data.dropna(subset=[value_column])
                    
                    if len(crop_ward_data) > 1 and crop_ward_data[value_column].std() > 0:
                        crop_ward_data['z_score'] = np.abs(stats.zscore(crop_ward_data[value_column]))
                        ward_outliers = crop_ward_data[crop_ward_data['z_score'] > z_threshold]
                        
                        if len(ward_outliers) > 0:
                            results['ward_level'][ward]['univariate'][value_column][crop] = {
                                'total_observations': len(crop_ward_data),
                                'outliers_count': len(ward_outliers),
                                'outliers': ward_outliers
                            }
                            
                            print(f"\nWard {ward}, Crop {crop}, Column {value_column}:")
                            print(f"Outliers detected: {len(ward_outliers)} out of {len(crop_ward_data)}")
                            if 'HouseHoldId' in ward_outliers.columns:
                                print(ward_outliers[[value_column, 'z_score', 'HouseHoldId']].head(3))
                            else:
                                print(ward_outliers[[value_column, 'z_score']].head(3))
                            if len(ward_outliers) > 3:
                                print(f"...and {len(ward_outliers)-3} more outliers")
    
    # 6. Generate summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    total_univariate_outliers = 0
    for column in value_columns:
        if column in results['univariate']:
            column_outliers = sum(results['univariate'][column].get(crop, {}).get('outliers_count', 0) 
                                 for crop in results['univariate'][column])
            total_univariate_outliers += column_outliers
            print(f"Column {column}: {column_outliers} univariate outliers detected")
    
    print(f"\nTotal univariate outliers across all columns: {total_univariate_outliers}")
    
    if isinstance(results['multivariate'], dict) and 'outliers_count' in results['multivariate']:
        print(f"Total multivariate outliers: {results['multivariate']['outliers_count']}")
    
    return results

# Usage:
#results = detect_outliers_crop_production(
#     crop_df, 
#     id_column='CropId',
#     value_columns=['AcresPlantedInLastFourWks', 'AcresHarvestedInLastFourWks', 
#                                             'KgsHarvestedInLastFourWks', 'OwnProductionStockInKg', 'KgsSoldInLastFourWks', 'PricePerKg'],
#     z_threshold=2.5
# )

#Livestock production
query = """
    SELECT hh_livestock_production_by_species.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_livestock_production_by_species.AnimalId,hh_livestock_production_by_species.NumberKeptToday,hh_livestock_production_by_species.NumberBornInLastFourWeeks,hh_livestock_production_by_species.NumberPurchasedInLastFourWeeks,hh_livestock_production_by_species.NumberSoldInLastFourWeeks,hh_livestock_production_by_species.AveragePricePerAnimalSold,hh_livestock_production_by_species.NumberDiedDuringLastFourWeeks
    FROM (hh_livestock_production_by_species
          LEFT JOIN hha_questionnaire_sessions ON (hh_livestock_production_by_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""

livestock_df = pd.read_sql(query, conn)

#conn.close()

# Create a date field from month and year
#livestock_df['date'] = pd.to_datetime(livestock_df['month'].astype(str) + ' ' + livestock_df['year'].astype(str), format='%B %Y')

#livestock_df

#Livestock production

def detect_outliers_livestock_production(livestock_df, id_column='AnimalId', 
                               value_columns=['NumberKeptToday', 'NumberBornInLastFourWeeks', 
                                             'NumberPurchasedInLastFourWeeks', 'NumberSoldInLastFourWeeks', 'AveragePricePerAnimalSold', 'NumberDiedDuringLastFourWeeks'],
                               z_threshold=2.5):
    """
    Comprehensive outlier detection for Livestock production data with multiple measurements
    Handles both univariate and multivariate outliers across multiple value columns
    Adapts to data size and checks for zero variance
    
    Parameters:
    -----------
    livestock_df : pandas DataFrame
        Input data with columns WardId, CountyId, AnimalId, and multiple measurement columns
    id_column : str
        Name of the Livestock ID column
    value_columns : list
        List of measurement column names to analyze
    z_threshold : float
        Z-score threshold for univariate outlier detection
    """
    results = {
        'dataset_info': {},
        'column_stats': {},
        'univariate': {col: {} for col in value_columns},
        'multivariate': {},
        'ward_level': {}
    }
    
    print("\nLIVESTOCK PRODUCTION OUTLIER ANALYSIS")
    print("=" * 50)    
    print("Starting outlier analysis...")

    
    # 1. Dataset Overview
    results['dataset_info'] = {
        'total_records': len(livestock_df),
        'unique_wards': livestock_df['WardId'].nunique(),
        'unique_Livestock': livestock_df[id_column].nunique(),
        'unique_households': livestock_df['HouseHoldId'].nunique() if 'HouseHoldId' in livestock_df.columns else 'N/A'
    }
    
    print("\nDATASET INFORMATION")
    print("=" * 50)
    for key, value in results['dataset_info'].items():
        print(f"{key}: {value}")
    
    # 2. Check stats for each column
    print("\nCOLUMN STATISTICS")
    print("=" * 50)
    
    for column in value_columns:
        # Skip if column doesn't exist in dataframe
        if column not in livestock_df.columns:
            print(f"Column {column} not found in dataframe. Skipping.")
            continue
            
        column_stats = {
            'mean': livestock_df[column].mean(),
            'median': livestock_df[column].median(),
            'std': livestock_df[column].std(),
            'min': livestock_df[column].min(),
            'max': livestock_df[column].max(),
            'null_count': livestock_df[column].isnull().sum(),
            'zero_count': (livestock_df[column] == 0).sum()
        }
        results['column_stats'][column] = column_stats
        
        print(f"\nColumn: {column}")
        for stat_name, stat_value in column_stats.items():
            print(f"{stat_name}: {stat_value}")
    
    # 3. Process each measurement column
    for value_column in value_columns:
        if value_column not in livestock_df.columns:
            continue
        
        print(f"\n\nANALYZING COLUMN: {value_column}")
        print("=" * 50)
        
        # 3.1 Check for zero variance Livestock in this column
        print("\nCHECKING VARIANCE")
        print("-" * 40)
        
        zero_var_Livestock = []
        valid_Livestock = []
        
        for Livestock in livestock_df[id_column].unique():
            Livestock_data = livestock_df[livestock_df[id_column] == Livestock][value_column]
            # Skip if all values are null
            if Livestock_data.isnull().all():
                print(f"Livestock {Livestock}: All values are NULL for {value_column}")
                continue
                
            # Handle nulls by dropping them for variance check
            Livestock_data = Livestock_data.dropna()
            
            if len(Livestock_data) == 0 or Livestock_data.std() == 0:
                zero_var_Livestock.append(Livestock)
                if len(Livestock_data) > 0:
                    constant_value = Livestock_data.iloc[0]
                    print(f"Livestock {Livestock}: Zero variance (all values = {constant_value})")
                else:
                    print(f"Livestock {Livestock}: No valid data after removing nulls")
            else:
                valid_Livestock.append(Livestock)
                print(f"Livestock {Livestock}: Has variation")
        
        # 3.2 Univariate Outlier Detection for this column
        print("\nCOUNTY-LEVEL ANALYSIS")
        print("=" * 50)
        print(f"\nUNIVARIATE OUTLIER DETECTION FOR {value_column}")
        print("-" * 40)
        
        for Livestock in valid_Livestock:
            Livestock_data = livestock_df[livestock_df[id_column] == Livestock].copy()
            # Remove nulls for z-score calculation
            Livestock_data = Livestock_data.dropna(subset=[value_column])
            
            if len(Livestock_data) <= 1:
                print(f"Livestock {Livestock}: Insufficient data for z-score calculation")
                continue
                
            Livestock_data['z_score'] = np.abs(stats.zscore(Livestock_data[value_column]))
            outliers = Livestock_data[Livestock_data['z_score'] > z_threshold]
            
            results['univariate'][value_column][Livestock] = {
                'total_observations': len(Livestock_data),
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers)/len(Livestock_data)*100) if len(Livestock_data) > 0 else 0,
                'mean': Livestock_data[value_column].mean(),
                'std': Livestock_data[value_column].std(),
                'outliers': outliers
            }
            
            print(f"\nLivestock {Livestock}:")
            print(f"Total valid observations: {len(Livestock_data)}")
            print(f"Outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                print(f"Outlier percentage: {(len(outliers)/len(Livestock_data)*100):.2f}%")
                if 'WardId' in outliers.columns and 'HouseHoldId' in outliers.columns:
                    outlier_info = outliers[[value_column, 'z_score', 'WardId', 'HouseHoldId']]
                else:
                    outlier_info = outliers[[value_column, 'z_score']]
                print("\nOutlier details:")
                print(outlier_info.head(5))  # Show first 5 outliers only
                if len(outliers) > 5:
                    print(f"...and {len(outliers)-5} more outliers")
    
    # 4. Multivariate Outlier Detection across all value columns
    print("\nMULTIVARIATE OUTLIER DETECTION")
    print("=" * 50)
    
    # Filter to columns that exist in the dataframe
    available_columns = [col for col in value_columns if col in livestock_df.columns]
    
    if len(available_columns) >= 2:  # Need at least 2 variables for multivariate analysis
        try:
            # Create a copy for multivariate analysis
            mv_livestock_df = livestock_df.copy()
            
            # For multivariate analysis, we need to handle missing values
            # Strategy: Impute with median for each Livestock
            for Livestock in livestock_df[id_column].unique():
                Livestock_mask = mv_livestock_df[id_column] == Livestock
                for col in available_columns:
                    median_val = mv_livestock_df.loc[Livestock_mask, col].median()
                    # Only fill if median is not NaN
                    if pd.notna(median_val):
                        mv_livestock_df.loc[Livestock_mask, col] = mv_livestock_df.loc[Livestock_mask, col].fillna(median_val)
            
            # Drop any remaining rows with NaN after imputation
            mv_livestock_df = mv_livestock_df.dropna(subset=available_columns)
            
            if len(mv_livestock_df) < 10:  # Not enough data for meaningful multivariate analysis
                print("Insufficient data for multivariate analysis after handling missing values")
                results['multivariate'] = {'error': 'Insufficient data after handling missing values'}
            else:
                # Group by IDs for household-level analysis
                id_columns = ['HouseHoldId', 'WardId', 'CountyId'] if 'HouseHoldId' in mv_livestock_df.columns else ['WardId', 'CountyId']
                available_id_columns = [col for col in id_columns if col in mv_livestock_df.columns]
                
                if not available_id_columns:
                    print("No ID columns available for grouping")
                    results['multivariate'] = {'error': 'No ID columns available for grouping'}
                else:
                    # Calculate Mahalanobis distance
                    X = mv_livestock_df[available_columns].values
                    mean_vec = np.mean(X, axis=0)
                    cov_matrix = np.cov(X, rowvar=False)
                    
                    # Handle potential singularity with pseudo-inverse
                    try:
                        inv_covmat = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
                        inv_covmat = np.linalg.pinv(cov_matrix)
                    
                    mv_livestock_df['mahalanobis_dist'] = np.zeros(len(mv_livestock_df))
                    for idx in range(len(X)):
                        diff = X[idx] - mean_vec
                        mv_livestock_df.loc[mv_livestock_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))
                    
                    # Chi-square threshold for multivariate outliers
                    threshold = chi2.ppf(0.975, livestock_df=len(available_columns))
                    mv_livestock_df['multivar_outlier'] = mv_livestock_df['mahalanobis_dist'] > threshold
                    
                    multivar_outliers = mv_livestock_df[mv_livestock_df['multivar_outlier']]
                    
                    results['multivariate'] = {
                        'total_observations': len(mv_livestock_df),
                        'outliers_count': len(multivar_outliers),
                        'outliers_percentage': (len(multivar_outliers)/len(mv_livestock_df)*100),
                        'mahalanobis_threshold': threshold,
                        'variables_used': available_columns,
                        'outliers': multivar_outliers
                    }
                    
                    print(f"\nMultivariate analysis results:")
                    print(f"Total observations: {len(mv_livestock_df)}")
                    print(f"Variables used: {available_columns}")
                    print(f"Outliers detected: {len(multivar_outliers)} ({(len(multivar_outliers)/len(mv_livestock_df)*100):.2f}%)")
                    if len(multivar_outliers) > 0:
                        display_columns = available_id_columns + ['mahalanobis_dist'] + available_columns
                        print("\nMultivariate outlier details (first 5):")
                        print(multivar_outliers[display_columns].head(5))
                        if len(multivar_outliers) > 5:
                            print(f"...and {len(multivar_outliers)-5} more outliers")
        
        except Exception as e:
            print(f"Note: Multivariate analysis could not be completed: {str(e)}")
            results['multivariate'] = {'error': str(e)}
    else:
        print(f"Insufficient variables for multivariate analysis. Need at least 2, found {len(available_columns)}")
        results['multivariate'] = {'error': 'Insufficient variables with variation'}
    
    # 5. Ward-level Analysis
    if 'WardId' in livestock_df.columns:
        print("\nWARD-LEVEL ANALYSIS")
        print("=" * 50)
        
        for ward in livestock_df['WardId'].unique():
            ward_data = livestock_df[livestock_df['WardId'] == ward]
            results['ward_level'][ward] = {
                'univariate': {col: {} for col in value_columns},
            }
            
            print(f"\nWard {ward}:")
            
            # Ward-level univariate outliers for each column
            for value_column in value_columns:
                if value_column not in livestock_df.columns:
                    continue
                    
                for Livestock in livestock_df[id_column].unique():
                    Livestock_ward_data = ward_data[ward_data[id_column] == Livestock].copy()
                    
                    # Skip if not enough data
                    if len(Livestock_ward_data) <= 1 or Livestock_ward_data[value_column].isnull().all():
                        continue
                        
                    # Remove nulls
                    Livestock_ward_data = Livestock_ward_data.dropna(subset=[value_column])
                    
                    if len(Livestock_ward_data) > 1 and Livestock_ward_data[value_column].std() > 0:
                        Livestock_ward_data['z_score'] = np.abs(stats.zscore(Livestock_ward_data[value_column]))
                        ward_outliers = Livestock_ward_data[Livestock_ward_data['z_score'] > z_threshold]
                        
                        if len(ward_outliers) > 0:
                            results['ward_level'][ward]['univariate'][value_column][Livestock] = {
                                'total_observations': len(Livestock_ward_data),
                                'outliers_count': len(ward_outliers),
                                'outliers': ward_outliers
                            }
                            
                            print(f"\nWard {ward}, Livestock {Livestock}, Column {value_column}:")
                            print(f"Outliers detected: {len(ward_outliers)} out of {len(Livestock_ward_data)}")
                            if 'HouseHoldId' in ward_outliers.columns:
                                print(ward_outliers[[value_column, 'z_score', 'HouseHoldId']].head(3))
                            else:
                                print(ward_outliers[[value_column, 'z_score']].head(3))
                            if len(ward_outliers) > 3:
                                print(f"...and {len(ward_outliers)-3} more outliers")
    
    # 6. Generate summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    total_univariate_outliers = 0
    for column in value_columns:
        if column in results['univariate']:
            column_outliers = sum(results['univariate'][column].get(Livestock, {}).get('outliers_count', 0) 
                                 for Livestock in results['univariate'][column])
            total_univariate_outliers += column_outliers
            print(f"Column {column}: {column_outliers} univariate outliers detected")
    
    print(f"\nTotal univariate outliers across all columns: {total_univariate_outliers}")
    
    if isinstance(results['multivariate'], dict) and 'outliers_count' in results['multivariate']:
        print(f"Total multivariate outliers: {results['multivariate']['outliers_count']}")
    
    return results

# Usage:
#results = detect_outliers_livestock_production(
#     livestock_df, 
#     id_column='AnimalId',
#     value_columns=['NumberKeptToday', 'NumberBornInLastFourWeeks', 
#                                             'NumberPurchasedInLastFourWeeks', 'NumberSoldInLastFourWeeks', 'AveragePricePerAnimalSold', 'NumberDiedDuringLastFourWeeks'],
#     z_threshold=2.5
# )

#Milk production
query = """
    SELECT hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_livestock_milk_production_per_species.AnimalId,hh_livestock_milk_production_per_species.DailyQntyMilkedInLtrs,hh_livestock_milk_production_per_species.DailyQntyConsumedInLtrs,hh_livestock_milk_production_per_species.DailyQntySoldInLtrs,hh_livestock_milk_production_per_species.PricePerLtr
    FROM (hh_livestock_milk_production_per_species
          LEFT JOIN hha_questionnaire_sessions ON (hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""

milk_df = pd.read_sql(query, conn)

#conn.close()

# Create a date field from month and year
#milk_df['date'] = pd.to_datetime(milk_df['month'].astype(str) + ' ' + milk_df['year'].astype(str), format='%B %Y')

#milk_df


#Milk production
def detect_outliers_milk(milk_df, id_column='AnimalId', 
                               value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 
                                             'DailyQntySoldInLtrs', 'PricePerLtr'],
                               z_threshold=2.5):
    """
    Comprehensive outlier detection for Milk production data with multiple measurements
    Handles both univariate and multivariate outliers across multiple value columns
    Adapts to data size and checks for zero variance
    
    Parameters:
    -----------
    milk_df : pandas DataFrame
        Input data with columns WardId, CountyId, AnimalId, and multiple measurement columns
    id_column : str
        Name of the Livestock ID column
    value_columns : list
        List of measurement column names to analyze
    z_threshold : float
        Z-score threshold for univariate outlier detection
    """
    results = {
        'dataset_info': {},
        'column_stats': {},
        'univariate': {col: {} for col in value_columns},
        'multivariate': {},
        'ward_level': {}
    }
    
    print("\nMILK PRODUCTION OUTLIER ANALYSIS")
    print("=" * 50)    
    print("Starting outlier analysis...")
    
    # 1. Dataset Overview
    results['dataset_info'] = {
        'total_records': len(milk_df),
        'unique_wards': milk_df['WardId'].nunique(),
        'unique_Livestock': milk_df[id_column].nunique(),
        'unique_households': milk_df['HouseHoldId'].nunique() if 'HouseHoldId' in milk_df.columns else 'N/A'
    }
    
    print("\nDATASET INFORMATION")
    print("=" * 50)
    for key, value in results['dataset_info'].items():
        print(f"{key}: {value}")
    
    # 2. Check stats for each column
    print("\nCOLUMN STATISTICS")
    print("=" * 50)
    
    for column in value_columns:
        # Skip if column doesn't exist in dataframe
        if column not in milk_df.columns:
            print(f"Column {column} not found in dataframe. Skipping.")
            continue
            
        column_stats = {
            'mean': milk_df[column].mean(),
            'median': milk_df[column].median(),
            'std': milk_df[column].std(),
            'min': milk_df[column].min(),
            'max': milk_df[column].max(),
            'null_count': milk_df[column].isnull().sum(),
            'zero_count': (milk_df[column] == 0).sum()
        }
        results['column_stats'][column] = column_stats
        
        print(f"\nColumn: {column}")
        for stat_name, stat_value in column_stats.items():
            print(f"{stat_name}: {stat_value}")
    
    # 3. Process each measurement column
    for value_column in value_columns:
        if value_column not in milk_df.columns:
            continue
        
        print(f"\n\nANALYZING COLUMN: {value_column}")
        print("=" * 50)
        
        # 3.1 Check for zero variance Livestock in this column
        print("\nCHECKING VARIANCE")
        print("-" * 40)
        
        zero_var_Livestock = []
        valid_Livestock = []
        
        for Livestock in milk_df[id_column].unique():
            Livestock_data = milk_df[milk_df[id_column] == Livestock][value_column]
            # Skip if all values are null
            if Livestock_data.isnull().all():
                print(f"Livestock {Livestock}: All values are NULL for {value_column}")
                continue
                
            # Handle nulls by dropping them for variance check
            Livestock_data = Livestock_data.dropna()
            
            if len(Livestock_data) == 0 or Livestock_data.std() == 0:
                zero_var_Livestock.append(Livestock)
                if len(Livestock_data) > 0:
                    constant_value = Livestock_data.iloc[0]
                    print(f"Livestock {Livestock}: Zero variance (all values = {constant_value})")
                else:
                    print(f"Livestock {Livestock}: No valid data after removing nulls")
            else:
                valid_Livestock.append(Livestock)
                print(f"Livestock {Livestock}: Has variation")
        
        # 3.2 Univariate Outlier Detection for this column
        print(f"\nUNIVARIATE OUTLIER DETECTION FOR {value_column}")
        print("-" * 40)
        
        for Livestock in valid_Livestock:
            Livestock_data = milk_df[milk_df[id_column] == Livestock].copy()
            # Remove nulls for z-score calculation
            Livestock_data = Livestock_data.dropna(subset=[value_column])
            
            if len(Livestock_data) <= 1:
                print(f"Livestock {Livestock}: Insufficient data for z-score calculation")
                continue
                
            Livestock_data['z_score'] = np.abs(stats.zscore(Livestock_data[value_column]))
            outliers = Livestock_data[Livestock_data['z_score'] > z_threshold]
            
            results['univariate'][value_column][Livestock] = {
                'total_observations': len(Livestock_data),
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers)/len(Livestock_data)*100) if len(Livestock_data) > 0 else 0,
                'mean': Livestock_data[value_column].mean(),
                'std': Livestock_data[value_column].std(),
                'outliers': outliers
            }
            
            print(f"\nLivestock {Livestock}:")
            print(f"Total valid observations: {len(Livestock_data)}")
            print(f"Outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                print(f"Outlier percentage: {(len(outliers)/len(Livestock_data)*100):.2f}%")
                if 'WardId' in outliers.columns and 'HouseHoldId' in outliers.columns:
                    outlier_info = outliers[[value_column, 'z_score', 'WardId', 'HouseHoldId']]
                else:
                    outlier_info = outliers[[value_column, 'z_score']]
                print("\nOutlier details:")
                print(outlier_info.head(5))  # Show first 5 outliers only
                if len(outliers) > 5:
                    print(f"...and {len(outliers)-5} more outliers")
    
    # 4. Multivariate Outlier Detection across all value columns
    print("\nMULTIVARIATE OUTLIER DETECTION")
    print("=" * 50)
    
    # Filter to columns that exist in the dataframe
    available_columns = [col for col in value_columns if col in milk_df.columns]
    
    if len(available_columns) >= 2:  # Need at least 2 variables for multivariate analysis
        try:
            # Create a copy for multivariate analysis
            mv_milk_df = milk_df.copy()
            
            # For multivariate analysis, we need to handle missing values
            # Strategy: Impute with median for each Livestock
            for Livestock in milk_df[id_column].unique():
                Livestock_mask = mv_milk_df[id_column] == Livestock
                for col in available_columns:
                    median_val = mv_milk_df.loc[Livestock_mask, col].median()
                    # Only fill if median is not NaN
                    if pd.notna(median_val):
                        mv_milk_df.loc[Livestock_mask, col] = mv_milk_df.loc[Livestock_mask, col].fillna(median_val)
            
            # Drop any remaining rows with NaN after imputation
            mv_milk_df = mv_milk_df.dropna(subset=available_columns)
            
            if len(mv_milk_df) < 10:  # Not enough data for meaningful multivariate analysis
                print("Insufficient data for multivariate analysis after handling missing values")
                results['multivariate'] = {'error': 'Insufficient data after handling missing values'}
            else:
                # Group by IDs for household-level analysis
                id_columns = ['HouseHoldId', 'WardId', 'CountyId'] if 'HouseHoldId' in mv_milk_df.columns else ['WardId', 'CountyId']
                available_id_columns = [col for col in id_columns if col in mv_milk_df.columns]
                
                if not available_id_columns:
                    print("No ID columns available for grouping")
                    results['multivariate'] = {'error': 'No ID columns available for grouping'}
                else:
                    # Calculate Mahalanobis distance
                    X = mv_milk_df[available_columns].values
                    mean_vec = np.mean(X, axis=0)
                    cov_matrix = np.cov(X, rowvar=False)
                    
                    # Handle potential singularity with pseudo-inverse
                    try:
                        inv_covmat = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
                        inv_covmat = np.linalg.pinv(cov_matrix)
                    
                    mv_milk_df['mahalanobis_dist'] = np.zeros(len(mv_milk_df))
                    for idx in range(len(X)):
                        diff = X[idx] - mean_vec
                        mv_milk_df.loc[mv_milk_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))
                    
                    # Chi-square threshold for multivariate outliers
                    threshold = chi2.ppf(0.975, milk_df=len(available_columns))
                    mv_milk_df['multivar_outlier'] = mv_milk_df['mahalanobis_dist'] > threshold
                    
                    multivar_outliers = mv_milk_df[mv_milk_df['multivar_outlier']]
                    
                    results['multivariate'] = {
                        'total_observations': len(mv_milk_df),
                        'outliers_count': len(multivar_outliers),
                        'outliers_percentage': (len(multivar_outliers)/len(mv_milk_df)*100),
                        'mahalanobis_threshold': threshold,
                        'variables_used': available_columns,
                        'outliers': multivar_outliers
                    }
                    
                    print(f"\nMultivariate analysis results:")
                    print(f"Total observations: {len(mv_milk_df)}")
                    print(f"Variables used: {available_columns}")
                    print(f"Outliers detected: {len(multivar_outliers)} ({(len(multivar_outliers)/len(mv_milk_df)*100):.2f}%)")
                    if len(multivar_outliers) > 0:
                        display_columns = available_id_columns + ['mahalanobis_dist'] + available_columns
                        print("\nMultivariate outlier details (first 5):")
                        print(multivar_outliers[display_columns].head(5))
                        if len(multivar_outliers) > 5:
                            print(f"...and {len(multivar_outliers)-5} more outliers")
        
        except Exception as e:
            print(f"Note: Multivariate analysis could not be completed: {str(e)}")
            results['multivariate'] = {'error': str(e)}
    else:
        print(f"Insufficient variables for multivariate analysis. Need at least 2, found {len(available_columns)}")
        results['multivariate'] = {'error': 'Insufficient variables with variation'}
    
    # 5. Ward-level Analysis
    if 'WardId' in milk_df.columns:
        print("\nWARD-LEVEL ANALYSIS")
        print("=" * 50)
        
        for ward in milk_df['WardId'].unique():
            ward_data = milk_df[milk_df['WardId'] == ward]
            results['ward_level'][ward] = {
                'univariate': {col: {} for col in value_columns},
            }
            
            print(f"\nWard {ward}:")
            
            # Ward-level univariate outliers for each column
            for value_column in value_columns:
                if value_column not in milk_df.columns:
                    continue
                    
                for Livestock in milk_df[id_column].unique():
                    Livestock_ward_data = ward_data[ward_data[id_column] == Livestock].copy()
                    
                    # Skip if not enough data
                    if len(Livestock_ward_data) <= 1 or Livestock_ward_data[value_column].isnull().all():
                        continue
                        
                    # Remove nulls
                    Livestock_ward_data = Livestock_ward_data.dropna(subset=[value_column])
                    
                    if len(Livestock_ward_data) > 1 and Livestock_ward_data[value_column].std() > 0:
                        Livestock_ward_data['z_score'] = np.abs(stats.zscore(Livestock_ward_data[value_column]))
                        ward_outliers = Livestock_ward_data[Livestock_ward_data['z_score'] > z_threshold]
                        
                        if len(ward_outliers) > 0:
                            results['ward_level'][ward]['univariate'][value_column][Livestock] = {
                                'total_observations': len(Livestock_ward_data),
                                'outliers_count': len(ward_outliers),
                                'outliers': ward_outliers
                            }
                            
                            print(f"\nWard {ward}, Livestock {Livestock}, Column {value_column}:")
                            print(f"Outliers detected: {len(ward_outliers)} out of {len(Livestock_ward_data)}")
                            if 'HouseHoldId' in ward_outliers.columns:
                                print(ward_outliers[[value_column, 'z_score', 'HouseHoldId']].head(3))
                            else:
                                print(ward_outliers[[value_column, 'z_score']].head(3))
                            if len(ward_outliers) > 3:
                                print(f"...and {len(ward_outliers)-3} more outliers")
    
    # 6. Generate summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    total_univariate_outliers = 0
    for column in value_columns:
        if column in results['univariate']:
            column_outliers = sum(results['univariate'][column].get(Livestock, {}).get('outliers_count', 0) 
                                 for Livestock in results['univariate'][column])
            total_univariate_outliers += column_outliers
            print(f"Column {column}: {column_outliers} univariate outliers detected")
    
    print(f"\nTotal univariate outliers across all columns: {total_univariate_outliers}")
    
    if isinstance(results['multivariate'], dict) and 'outliers_count' in results['multivariate']:
        print(f"Total multivariate outliers: {results['multivariate']['outliers_count']}")
    
    return results

# Usage:
#results = detect_outliers_milk(
#     milk_df, 
#     id_column='AnimalId',
#     value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 
#                                             'DailyQntySoldInLtrs', 'PricePerLtr'],
#     z_threshold=2.5
# )

query = """
    SELECT hh_consumption_coping_strategies.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_consumption_coping_strategies.CopyingStrategyId,hh_consumption_coping_strategies.NumOfCopingDays
    FROM (hh_consumption_coping_strategies
          LEFT JOIN hha_questionnaire_sessions ON (hh_consumption_coping_strategies.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""

coping_df = pd.read_sql(query, conn)

#conn.close()

# Create a date field from month and year
#coping_df['date'] = pd.to_datetime(coping_df['month'].astype(str) + ' ' + coping_df['year'].astype(str), format='%B %Y')

#coping_df

#Coping strategies
def detect_outliers_Copying_Strategies(coping_df, id_column='CopyingStrategyId', 
                               value_columns=['NumOfCopingDays'],
                               z_threshold=2.5):
    """
    Comprehensive outlier detection for Coping_strategy data with multiple measurements
    Handles both univariate and multivariate outliers across multiple value columns
    Adapts to data size and checks for zero variance
    
    Parameters:
    -----------
    coping_df : pandas DataFrame
        Input data with columns WardId, CountyId, CopyingStrategyId, and multiple measurement columns
    id_column : str
        Name of the Coping_strategy ID column
    value_columns : list
        List of measurement column names to analyze
    z_threshold : float
        Z-score threshold for univariate outlier detection
    """
    results = {
        'dataset_info': {},
        'column_stats': {},
        'univariate': {col: {} for col in value_columns},
        'multivariate': {},
        'ward_level': {}
    }
    
    print("\nCOPING STRATEGIES OUTLIER ANALYSIS")
    print("=" * 50)    
    print("Starting outlier analysis...")
    
    # 1. Dataset Overview
    results['dataset_info'] = {
        'total_records': len(coping_df),
        'unique_wards': coping_df['WardId'].nunique(),
        'unique_Coping_strategies': coping_df[id_column].nunique(),
        'unique_households': coping_df['HouseHoldId'].nunique() if 'HouseHoldId' in coping_df.columns else 'N/A'
    }
    
    print("\nDATASET INFORMATION")
    print("=" * 50)
    for key, value in results['dataset_info'].items():
        print(f"{key}: {value}")
    
    # 2. Check stats for each column
    print("\nCOLUMN STATISTICS")
    print("=" * 50)
    
    for column in value_columns:
        # Skip if column doesn't exist in dataframe
        if column not in coping_df.columns:
            print(f"Column {column} not found in dataframe. Skipping.")
            continue
            
        column_stats = {
            'mean': coping_df[column].mean(),
            'median': coping_df[column].median(),
            'std': coping_df[column].std(),
            'min': coping_df[column].min(),
            'max': coping_df[column].max(),
            'null_count': coping_df[column].isnull().sum(),
            'zero_count': (coping_df[column] == 0).sum()
        }
        results['column_stats'][column] = column_stats
        
        print(f"\nColumn: {column}")
        for stat_name, stat_value in column_stats.items():
            print(f"{stat_name}: {stat_value}")
    
    # 3. Process each measurement column
    for value_column in value_columns:
        if value_column not in coping_df.columns:
            continue
        
        print(f"\n\nANALYZING COLUMN: {value_column}")
        print("=" * 50)
        
        # 3.1 Check for zero variance Coping_strategies in this column
        print("\nCHECKING VARIANCE")
        print("-" * 40)
        
        zero_var_Coping_strategies = []
        valid_Coping_strategies = []
        
        for Coping_strategy in coping_df[id_column].unique():
            Coping_strategy_data = coping_df[coping_df[id_column] == Coping_strategy][value_column]
            # Skip if all values are null
            if Coping_strategy_data.isnull().all():
                print(f"Coping_strategy {Coping_strategy}: All values are NULL for {value_column}")
                continue
                
            # Handle nulls by dropping them for variance check
            Coping_strategy_data = Coping_strategy_data.dropna()
            
            if len(Coping_strategy_data) == 0 or Coping_strategy_data.std() == 0:
                zero_var_Coping_strategies.append(Coping_strategy)
                if len(Coping_strategy_data) > 0:
                    constant_value = Coping_strategy_data.iloc[0]
                    print(f"Coping_strategy {Coping_strategy}: Zero variance (all values = {constant_value})")
                else:
                    print(f"Coping_strategy {Coping_strategy}: No valid data after removing nulls")
            else:
                valid_Coping_strategies.append(Coping_strategy)
                print(f"Coping_strategy {Coping_strategy}: Has variation")
        
        # 3.2 Univariate Outlier Detection for this column
        print("\nCOUNTY-LEVEL ANALYSIS")
        print("=" * 50)
        print(f"\nUNIVARIATE OUTLIER DETECTION FOR {value_column}")
        print("-" * 40)
        
        for Coping_strategy in valid_Coping_strategies:
            Coping_strategy_data = coping_df[coping_df[id_column] == Coping_strategy].copy()
            # Remove nulls for z-score calculation
            Coping_strategy_data = Coping_strategy_data.dropna(subset=[value_column])
            
            if len(Coping_strategy_data) <= 1:
                print(f"Coping_strategy {Coping_strategy}: Insufficient data for z-score calculation")
                continue
                
            Coping_strategy_data['z_score'] = np.abs(stats.zscore(Coping_strategy_data[value_column]))
            outliers = Coping_strategy_data[Coping_strategy_data['z_score'] > z_threshold]
            
            results['univariate'][value_column][Coping_strategy] = {
                'total_observations': len(Coping_strategy_data),
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers)/len(Coping_strategy_data)*100) if len(Coping_strategy_data) > 0 else 0,
                'mean': Coping_strategy_data[value_column].mean(),
                'std': Coping_strategy_data[value_column].std(),
                'outliers': outliers
            }
            
            print(f"\nCoping_strategy {Coping_strategy}:")
            print(f"Total valid observations: {len(Coping_strategy_data)}")
            print(f"Outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                print(f"Outlier percentage: {(len(outliers)/len(Coping_strategy_data)*100):.2f}%")
                if 'WardId' in outliers.columns and 'HouseHoldId' in outliers.columns:
                    outlier_info = outliers[[value_column, 'z_score', 'WardId', 'HouseHoldId']]
                else:
                    outlier_info = outliers[[value_column, 'z_score']]
                print("\nOutlier details:")
                print(outlier_info.head(5))  # Show first 5 outliers only
                if len(outliers) > 5:
                    print(f"...and {len(outliers)-5} more outliers")
    
    # 4. Multivariate Outlier Detection across all value columns
    print("\nMULTIVARIATE OUTLIER DETECTION")
    print("=" * 50)
    
    # Filter to columns that exist in the dataframe
    available_columns = [col for col in value_columns if col in coping_df.columns]
    
    if len(available_columns) >= 2:  # Need at least 2 variables for multivariate analysis
        try:
            # Create a copy for multivariate analysis
            mv_coping_df = coping_df.copy()
            
            # For multivariate analysis, we need to handle missing values
            # Strategy: Impute with median for each Coping_strategy
            for Coping_strategy in coping_df[id_column].unique():
                Coping_strategy_mask = mv_coping_df[id_column] == Coping_strategy
                for col in available_columns:
                    median_val = mv_coping_df.loc[Coping_strategy_mask, col].median()
                    # Only fill if median is not NaN
                    if pd.notna(median_val):
                        mv_coping_df.loc[Coping_strategy_mask, col] = mv_coping_df.loc[Coping_strategy_mask, col].fillna(median_val)
            
            # Drop any remaining rows with NaN after imputation
            mv_coping_df = mv_coping_df.dropna(subset=available_columns)
            
            if len(mv_coping_df) < 10:  # Not enough data for meaningful multivariate analysis
                print("Insufficient data for multivariate analysis after handling missing values")
                results['multivariate'] = {'error': 'Insufficient data after handling missing values'}
            else:
                # Group by IDs for household-level analysis
                id_columns = ['HouseHoldId', 'WardId', 'CountyId'] if 'HouseHoldId' in mv_coping_df.columns else ['WardId', 'CountyId']
                available_id_columns = [col for col in id_columns if col in mv_coping_df.columns]
                
                if not available_id_columns:
                    print("No ID columns available for grouping")
                    results['multivariate'] = {'error': 'No ID columns available for grouping'}
                else:
                    # Calculate Mahalanobis distance
                    X = mv_coping_df[available_columns].values
                    mean_vec = np.mean(X, axis=0)
                    cov_matrix = np.cov(X, rowvar=False)
                    
                    # Handle potential singularity with pseudo-inverse
                    try:
                        inv_covmat = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
                        inv_covmat = np.linalg.pinv(cov_matrix)
                    
                    mv_coping_df['mahalanobis_dist'] = np.zeros(len(mv_coping_df))
                    for idx in range(len(X)):
                        diff = X[idx] - mean_vec
                        mv_coping_df.loc[mv_coping_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))
                    
                    # Chi-square threshold for multivariate outliers
                    threshold = chi2.ppf(0.975, coping_df=len(available_columns))
                    mv_coping_df['multivar_outlier'] = mv_coping_df['mahalanobis_dist'] > threshold
                    
                    multivar_outliers = mv_coping_df[mv_coping_df['multivar_outlier']]
                    
                    results['multivariate'] = {
                        'total_observations': len(mv_coping_df),
                        'outliers_count': len(multivar_outliers),
                        'outliers_percentage': (len(multivar_outliers)/len(mv_coping_df)*100),
                        'mahalanobis_threshold': threshold,
                        'variables_used': available_columns,
                        'outliers': multivar_outliers
                    }
                    
                    print(f"\nMultivariate analysis results:")
                    print(f"Total observations: {len(mv_coping_df)}")
                    print(f"Variables used: {available_columns}")
                    print(f"Outliers detected: {len(multivar_outliers)} ({(len(multivar_outliers)/len(mv_coping_df)*100):.2f}%)")
                    if len(multivar_outliers) > 0:
                        display_columns = available_id_columns + ['mahalanobis_dist'] + available_columns
                        print("\nMultivariate outlier details (first 5):")
                        print(multivar_outliers[display_columns].head(5))
                        if len(multivar_outliers) > 5:
                            print(f"...and {len(multivar_outliers)-5} more outliers")
        
        except Exception as e:
            print(f"Note: Multivariate analysis could not be completed: {str(e)}")
            results['multivariate'] = {'error': str(e)}
    else:
        print(f"Insufficient variables for multivariate analysis. Need at least 2, found {len(available_columns)}")
        results['multivariate'] = {'error': 'Insufficient variables with variation'}
    
    # 5. Ward-level Analysis
    if 'WardId' in coping_df.columns:
        print("\nWARD-LEVEL ANALYSIS")
        print("=" * 50)
        
        for ward in coping_df['WardId'].unique():
            ward_data = coping_df[coping_df['WardId'] == ward]
            results['ward_level'][ward] = {
                'univariate': {col: {} for col in value_columns},
            }
            
            print(f"\nWard {ward}:")
            
            # Ward-level univariate outliers for each column
            for value_column in value_columns:
                if value_column not in coping_df.columns:
                    continue
                    
                for Coping_strategy in coping_df[id_column].unique():
                    Coping_strategy_ward_data = ward_data[ward_data[id_column] == Coping_strategy].copy()
                    
                    # Skip if not enough data
                    if len(Coping_strategy_ward_data) <= 1 or Coping_strategy_ward_data[value_column].isnull().all():
                        continue
                        
                    # Remove nulls
                    Coping_strategy_ward_data = Coping_strategy_ward_data.dropna(subset=[value_column])
                    
                    if len(Coping_strategy_ward_data) > 1 and Coping_strategy_ward_data[value_column].std() > 0:
                        Coping_strategy_ward_data['z_score'] = np.abs(stats.zscore(Coping_strategy_ward_data[value_column]))
                        ward_outliers = Coping_strategy_ward_data[Coping_strategy_ward_data['z_score'] > z_threshold]
                        
                        if len(ward_outliers) > 0:
                            results['ward_level'][ward]['univariate'][value_column][Coping_strategy] = {
                                'total_observations': len(Coping_strategy_ward_data),
                                'outliers_count': len(ward_outliers),
                                'outliers': ward_outliers
                            }
                            
                            print(f"\nWard {ward}, Coping_strategy {Coping_strategy}, Column {value_column}:")
                            print(f"Outliers detected: {len(ward_outliers)} out of {len(Coping_strategy_ward_data)}")
                            if 'HouseHoldId' in ward_outliers.columns:
                                print(ward_outliers[[value_column, 'z_score', 'HouseHoldId']].head(3))
                            else:
                                print(ward_outliers[[value_column, 'z_score']].head(3))
                            if len(ward_outliers) > 3:
                                print(f"...and {len(ward_outliers)-3} more outliers")
    
    # 6. Generate summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    total_univariate_outliers = 0
    for column in value_columns:
        if column in results['univariate']:
            column_outliers = sum(results['univariate'][column].get(Coping_strategy, {}).get('outliers_count', 0) 
                                 for Coping_strategy in results['univariate'][column])
            total_univariate_outliers += column_outliers
            print(f"Column {column}: {column_outliers} univariate outliers detected")
    
    print(f"\nTotal univariate outliers across all columns: {total_univariate_outliers}")
    
    if isinstance(results['multivariate'], dict) and 'outliers_count' in results['multivariate']:
        print(f"Total multivariate outliers: {results['multivariate']['outliers_count']}")
    
    return results

# Usage:
#results = detect_outliers_Copying_Strategies(
#     coping_df, 
#     id_column='CopyingStrategyId',
#     value_columns=['NumOfCopingDays'],
#     z_threshold=2.5
# )

#Food consumption
query = """
    SELECT hh_food_consumption.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_food_consumption.FoodTypeId,hh_food_consumption.NumDaysEaten
    FROM (hh_food_consumption
          LEFT JOIN hha_questionnaire_sessions ON (hh_food_consumption.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""

food_c_df = pd.read_sql(query, conn)

#conn.close()

# Create a date field from month and year
#food_c_df['date'] = pd.to_datetime(food_c_df['month'].astype(str) + ' ' + food_c_df['year'].astype(str), format='%B %Y')

#food_c_df

#Food Consumption

def detect_outliers_Food_Consumption(food_c_df, id_column='FoodTypeId', 
                               value_columns=['NumDaysEaten'],
                               z_threshold=2.5):
    """
    Comprehensive outlier detection for Food Consumption data with multiple measurements
    Handles both univariate and multivariate outliers across multiple value columns
    Adapts to data size and checks for zero variance
    
    Parameters:
    -----------
    food_c_df : pandas DataFrame
        Input data with columns WardId, CountyId, FoodTypeId, and multiple measurement columns
    id_column : str
        Name of the FoodType ID column
    value_columns : list
        List of measurement column names to analyze
    z_threshold : float
        Z-score threshold for univariate outlier detection
    """
    results = {
        'dataset_info': {},
        'column_stats': {},
        'univariate': {col: {} for col in value_columns},
        'multivariate': {},
        'ward_level': {}
    }
    print("\nFOOD CONSUMPTION OUTLIER ANALYSIS")
    print("=" * 50)    
    print("Starting outlier analysis...")

    
    # 1. Dataset Overview
    results['dataset_info'] = {
        'total_records': len(food_c_df),
        'unique_wards': food_c_df['WardId'].nunique(),
        'unique_FoodTypes': food_c_df[id_column].nunique(),
        'unique_households': food_c_df['HouseHoldId'].nunique() if 'HouseHoldId' in food_c_df.columns else 'N/A'
    }
    
    print("\nDATASET INFORMATION")
    print("=" * 50)
    for key, value in results['dataset_info'].items():
        print(f"{key}: {value}")
    
    # 2. Check stats for each column
    print("\nCOLUMN STATISTICS")
    print("=" * 50)
    
    for column in value_columns:
        # Skip if column doesn't exist in dataframe
        if column not in food_c_df.columns:
            print(f"Column {column} not found in dataframe. Skipping.")
            continue
            
        column_stats = {
            'mean': food_c_df[column].mean(),
            'median': food_c_df[column].median(),
            'std': food_c_df[column].std(),
            'min': food_c_df[column].min(),
            'max': food_c_df[column].max(),
            'null_count': food_c_df[column].isnull().sum(),
            'zero_count': (food_c_df[column] == 0).sum()
        }
        results['column_stats'][column] = column_stats
        
        print(f"\nColumn: {column}")
        for stat_name, stat_value in column_stats.items():
            print(f"{stat_name}: {stat_value}")
    
    # 3. Process each measurement column
    for value_column in value_columns:
        if value_column not in food_c_df.columns:
            continue
        
        print(f"\n\nANALYZING COLUMN: {value_column}")
        print("=" * 50)
        
        # 3.1 Check for zero variance FoodTypes in this column
        print("\nCHECKING VARIANCE")
        print("-" * 40)
        
        zero_var_FoodTypes = []
        valid_FoodTypes = []
        
        for FoodType in food_c_df[id_column].unique():
            FoodType_data = food_c_df[food_c_df[id_column] == FoodType][value_column]
            # Skip if all values are null
            if FoodType_data.isnull().all():
                print(f"FoodType {FoodType}: All values are NULL for {value_column}")
                continue
                
            # Handle nulls by dropping them for variance check
            FoodType_data = FoodType_data.dropna()
            
            if len(FoodType_data) == 0 or FoodType_data.std() == 0:
                zero_var_FoodTypes.append(FoodType)
                if len(FoodType_data) > 0:
                    constant_value = FoodType_data.iloc[0]
                    print(f"FoodType {FoodType}: Zero variance (all values = {constant_value})")
                else:
                    print(f"FoodType {FoodType}: No valid data after removing nulls")
            else:
                valid_FoodTypes.append(FoodType)
                print(f"FoodType {FoodType}: Has variation")
        
        # 3.2 Univariate Outlier Detection for this column
        print("\nCOUNTY-LEVEL ANALYSIS")
        print("=" * 50)
        print(f"\nUNIVARIATE OUTLIER DETECTION FOR {value_column}")
        print("-" * 40)
        
        for FoodType in valid_FoodTypes:
            FoodType_data = food_c_df[food_c_df[id_column] == FoodType].copy()
            # Remove nulls for z-score calculation
            FoodType_data = FoodType_data.dropna(subset=[value_column])
            
            if len(FoodType_data) <= 1:
                print(f"FoodType {FoodType}: Insufficient data for z-score calculation")
                continue
                
            FoodType_data['z_score'] = np.abs(stats.zscore(FoodType_data[value_column]))
            outliers = FoodType_data[FoodType_data['z_score'] > z_threshold]
            
            results['univariate'][value_column][FoodType] = {
                'total_observations': len(FoodType_data),
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers)/len(FoodType_data)*100) if len(FoodType_data) > 0 else 0,
                'mean': FoodType_data[value_column].mean(),
                'std': FoodType_data[value_column].std(),
                'outliers': outliers
            }
            
            print(f"\nFoodType {FoodType}:")
            print(f"Total valid observations: {len(FoodType_data)}")
            print(f"Outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                print(f"Outlier percentage: {(len(outliers)/len(FoodType_data)*100):.2f}%")
                if 'WardId' in outliers.columns and 'HouseHoldId' in outliers.columns:
                    outlier_info = outliers[[value_column, 'z_score', 'WardId', 'HouseHoldId']]
                else:
                    outlier_info = outliers[[value_column, 'z_score']]
                print("\nOutlier details:")
                print(outlier_info.head(5))  # Show first 5 outliers only
                if len(outliers) > 5:
                    print(f"...and {len(outliers)-5} more outliers")
    
    # 4. Multivariate Outlier Detection across all value columns
    print("\nMULTIVARIATE OUTLIER DETECTION")
    print("=" * 50)
    
    # Filter to columns that exist in the dataframe
    available_columns = [col for col in value_columns if col in food_c_df.columns]
    
    if len(available_columns) >= 2:  # Need at least 2 variables for multivariate analysis
        try:
            # Create a copy for multivariate analysis
            mv_food_c_df = food_c_df.copy()
            
            # For multivariate analysis, we need to handle missing values
            # Strategy: Impute with median for each FoodType
            for FoodType in food_c_df[id_column].unique():
                FoodType_mask = mv_food_c_df[id_column] == FoodType
                for col in available_columns:
                    median_val = mv_food_c_df.loc[FoodType_mask, col].median()
                    # Only fill if median is not NaN
                    if pd.notna(median_val):
                        mv_food_c_df.loc[FoodType_mask, col] = mv_food_c_df.loc[FoodType_mask, col].fillna(median_val)
            
            # Drop any remaining rows with NaN after imputation
            mv_food_c_df = mv_food_c_df.dropna(subset=available_columns)
            
            if len(mv_food_c_df) < 10:  # Not enough data for meaningful multivariate analysis
                print("Insufficient data for multivariate analysis after handling missing values")
                results['multivariate'] = {'error': 'Insufficient data after handling missing values'}
            else:
                # Group by IDs for household-level analysis
                id_columns = ['HouseHoldId', 'WardId', 'CountyId'] if 'HouseHoldId' in mv_food_c_df.columns else ['WardId', 'CountyId']
                available_id_columns = [col for col in id_columns if col in mv_food_c_df.columns]
                
                if not available_id_columns:
                    print("No ID columns available for grouping")
                    results['multivariate'] = {'error': 'No ID columns available for grouping'}
                else:
                    # Calculate Mahalanobis distance
                    X = mv_food_c_df[available_columns].values
                    mean_vec = np.mean(X, axis=0)
                    cov_matrix = np.cov(X, rowvar=False)
                    
                    # Handle potential singularity with pseudo-inverse
                    try:
                        inv_covmat = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
                        inv_covmat = np.linalg.pinv(cov_matrix)
                    
                    mv_food_c_df['mahalanobis_dist'] = np.zeros(len(mv_food_c_df))
                    for idx in range(len(X)):
                        diff = X[idx] - mean_vec
                        mv_food_c_df.loc[mv_food_c_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))
                    
                    # Chi-square threshold for multivariate outliers
                    threshold = chi2.ppf(0.975, food_c_df=len(available_columns))
                    mv_food_c_df['multivar_outlier'] = mv_food_c_df['mahalanobis_dist'] > threshold
                    
                    multivar_outliers = mv_food_c_df[mv_food_c_df['multivar_outlier']]
                    
                    results['multivariate'] = {
                        'total_observations': len(mv_food_c_df),
                        'outliers_count': len(multivar_outliers),
                        'outliers_percentage': (len(multivar_outliers)/len(mv_food_c_df)*100),
                        'mahalanobis_threshold': threshold,
                        'variables_used': available_columns,
                        'outliers': multivar_outliers
                    }
                    
                    print(f"\nMultivariate analysis results:")
                    print(f"Total observations: {len(mv_food_c_df)}")
                    print(f"Variables used: {available_columns}")
                    print(f"Outliers detected: {len(multivar_outliers)} ({(len(multivar_outliers)/len(mv_food_c_df)*100):.2f}%)")
                    if len(multivar_outliers) > 0:
                        display_columns = available_id_columns + ['mahalanobis_dist'] + available_columns
                        print("\nMultivariate outlier details (first 5):")
                        print(multivar_outliers[display_columns].head(5))
                        if len(multivar_outliers) > 5:
                            print(f"...and {len(multivar_outliers)-5} more outliers")
        
        except Exception as e:
            print(f"Note: Multivariate analysis could not be completed: {str(e)}")
            results['multivariate'] = {'error': str(e)}
    else:
        print(f"Insufficient variables for multivariate analysis. Need at least 2, found {len(available_columns)}")
        results['multivariate'] = {'error': 'Insufficient variables with variation'}
    
    # 5. Ward-level Analysis
    if 'WardId' in food_c_df.columns:
        print("\nWARD-LEVEL ANALYSIS")
        print("=" * 50)
        
        for ward in food_c_df['WardId'].unique():
            ward_data = food_c_df[food_c_df['WardId'] == ward]
            results['ward_level'][ward] = {
                'univariate': {col: {} for col in value_columns},
            }
            
            print(f"\nWard {ward}:")
            
            # Ward-level univariate outliers for each column
            for value_column in value_columns:
                if value_column not in food_c_df.columns:
                    continue
                    
                for FoodType in food_c_df[id_column].unique():
                    FoodType_ward_data = ward_data[ward_data[id_column] == FoodType].copy()
                    
                    # Skip if not enough data
                    if len(FoodType_ward_data) <= 1 or FoodType_ward_data[value_column].isnull().all():
                        continue
                        
                    # Remove nulls
                    FoodType_ward_data = FoodType_ward_data.dropna(subset=[value_column])
                    
                    if len(FoodType_ward_data) > 1 and FoodType_ward_data[value_column].std() > 0:
                        FoodType_ward_data['z_score'] = np.abs(stats.zscore(FoodType_ward_data[value_column]))
                        ward_outliers = FoodType_ward_data[FoodType_ward_data['z_score'] > z_threshold]
                        
                        if len(ward_outliers) > 0:
                            results['ward_level'][ward]['univariate'][value_column][FoodType] = {
                                'total_observations': len(FoodType_ward_data),
                                'outliers_count': len(ward_outliers),
                                'outliers': ward_outliers
                            }
                            
                            print(f"\nWard {ward}, FoodType {FoodType}, Column {value_column}:")
                            print(f"Outliers detected: {len(ward_outliers)} out of {len(FoodType_ward_data)}")
                            if 'HouseHoldId' in ward_outliers.columns:
                                print(ward_outliers[[value_column, 'z_score', 'HouseHoldId']].head(3))
                            else:
                                print(ward_outliers[[value_column, 'z_score']].head(3))
                            if len(ward_outliers) > 3:
                                print(f"...and {len(ward_outliers)-3} more outliers")
    
    # 6. Generate summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    
    total_univariate_outliers = 0
    for column in value_columns:
        if column in results['univariate']:
            column_outliers = sum(results['univariate'][column].get(FoodType, {}).get('outliers_count', 0) 
                                 for FoodType in results['univariate'][column])
            total_univariate_outliers += column_outliers
            print(f"Column {column}: {column_outliers} univariate outliers detected")
    
    print(f"\nTotal univariate outliers across all columns: {total_univariate_outliers}")
    
    if isinstance(results['multivariate'], dict) and 'outliers_count' in results['multivariate']:
        print(f"Total multivariate outliers: {results['multivariate']['outliers_count']}")
    
    return results

# Usage:
#results = detect_outliers_Food_Consumption(
#     food_c_df, 
#     id_column='FoodTypeId',
#     value_columns=['NumDaysEaten'],
#     z_threshold=2.5
# )

#Muac
query = """
    SELECT hh_children.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_children.HhChildId,hh_children.AgeInMonths,hh_children.MuacInMillimeters
    FROM (hh_children
          LEFT JOIN hha_questionnaire_sessions ON (hh_children.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""

df = pd.read_sql(query, conn)

#conn.close()

# Create a date field from month and year
#df['date'] = pd.to_datetime(df['month'].astype(str) + ' ' + df['year'].astype(str), format='%B %Y')

#df

#Muac
def analyze_muac_data(df, variables=['AgeInMonths', 'MuacInMillimeters'], z_threshold=2.5):
    """
    Analyze muac data at both CountyId and WardId levels
    """
    results = {
        'CountyId': {'univariate': {}, 'multivariate': {}},
        'WardId': {'univariate': {}, 'multivariate': {}}
    }
    
    # Get CountyId name
    CountyId_name = df['CountyId'].iloc[0]
    
    # =============== CountyId LEVEL ANALYSIS ===============
    print(f"\nMUAC Outlier Analysis for {CountyId_name} CountyId")
    print("=" * 100)
    
    # CountyId level plot
    #plt.figure(figsize=(12, 6))
    #df_melted = df.melt(value_vars=variables,
#                        var_name='Variable', 
#                        value_name='Value')
#    
#    sns.boxplot(data=df_melted, x='Variable', y='Value')
 #   plt.title(f'AgeInMonths and MuacInMillimeters in {CountyId_name} CountyId')
  #  plt.xlabel('Variable')
   # plt.ylabel('Amount')
#    plt.xticks(rotation=45)
 #   plt.tight_layout()
  #  plt.show()
    
    # CountyId level multivariate analysis
    df = detect_multivariate_outliers_by_WardId(df, variables, WardId_column=None)
    
    # CountyId level univariate analysis
    for variable in variables:
        df = detect_univariate_outliers_by_WardId(df, variable, z_threshold, WardId_column=None)
        
        # Store univariate results
        outliers = df[df['is_outlier']]
        total_count = len(df)
        outliers_count = len(outliers)
        
        results['CountyId']['univariate'][variable] = {
            'outliers': outliers,
            'total_observations': total_count,
            'outliers_count': outliers_count,
            'outliers_percentage': (outliers_count/total_count*100),
            'mean': df[variable].mean(),
            'median': df[variable].median(),
            'std_dev': df[variable].std()
        }
    
    # Store CountyId multivariate results
    multivar_outliers = df[df['multivar_outlier']]
    results['CountyId']['multivariate'] = {
        'total_observations': len(df),
        'outliers_count': len(multivar_outliers),
        'outliers_percentage': (len(multivar_outliers)/len(df)*100),
        'outliers': multivar_outliers
    }
    
    # Print CountyId level statistics
    print("\nCountyId Level Statistics")
    print("-" * 50)
    
    # CountyId univariate statistics
    print("\nUnivariate Analysis:")
    print(f"{'Variable':<15} {'Mean':>10} {'Median':>10} {'Std Dev':>10} {'Outliers':>10} {'Total':>10}")
    print("-" * 70)
    
    for variable in variables:
        stats = results['CountyId']['univariate'][variable]
        print(f"{variable:<15} {stats['mean']:10.2f} {stats['median']:10.2f} "
              f"{stats['std_dev']:10.2f} {stats['outliers_count']:10d} {stats['total_observations']:10d}")
    
    # CountyId multivariate statistics
    print("\nMultivariate Analysis:")
    multivar_stats = results['CountyId']['multivariate']
    print(f"Total observations: {multivar_stats['total_observations']}")
    print(f"Multivariate outliers: {multivar_stats['outliers_count']}")
    print(f"Percentage multivariate outliers: {multivar_stats['outliers_percentage']:.1f}%")
    
    # Print CountyId level outlier details
    print("\nCountyId Level Outlier Details")
    print("-" * 50)
    
    print("\nUnivariate Outliers:")
    for variable in variables:
        outliers = results['CountyId']['univariate'][variable]['outliers']
        if not outliers.empty:
            print(f"\nOutliers for {variable}:")
            print(outliers[['DataCollectionExerciseId', 'HouseHoldId', 
                          'WardId', variable, 'z_score']].to_string())
    
    print("\nMultivariate Outliers:")
    if not multivar_outliers.empty:
        print(multivar_outliers[['DataCollectionExerciseId', 'HouseHoldId', 
                                'WardId'] + variables + ['mahalanobis_dist']].to_string())
    
    # =============== WardId LEVEL ANALYSIS ===============
    print("\n\nAnalysis by WardId")
    print("=" * 100)
    
    # WardId level plot
#    plt.figure(figsize=(15, 8))
#    df_melted = df.melt(id_vars=['WardId'], 
#                        value_vars=variables,
#                        var_name='Variable', 
#                        value_name='Value')
#    
#    sns.boxplot(data=df_melted, x='WardId', y='Value', hue='Variable')
#    plt.title(f'AgeInMonths and MuacInMillimeters by WardId in {CountyId_name} CountyId')
#    plt.xlabel('WardId')
#    plt.ylabel('Amount')
#    plt.xticks(rotation=45)
#    plt.legend(title='Variable')
#    plt.tight_layout()
#    plt.show()
    
    # Perform WardId-level analysis (using existing functions)
    df = detect_multivariate_outliers_by_WardId(df, variables)
    
    for variable in variables:
        df = detect_univariate_outliers_by_WardId(df, variable, z_threshold)
        results['WardId']['univariate'][variable] = {
            'outliers': df[df['is_outlier']],
            'WardId_stats': {}
        }
        
        for WardId in df['WardId'].unique():
            WardId_data = df[df['WardId'] == WardId]
            outliers_count = WardId_data['is_outlier'].sum()
            total_count = len(WardId_data)
            
            results['WardId']['univariate'][variable]['WardId_stats'][WardId] = {
                'total_observations': total_count,
                'outliers_count': outliers_count,
                'outliers_percentage': (outliers_count/total_count*100),
                'mean': WardId_data[variable].mean(),
                'median': WardId_data[variable].median(),
                'std_dev': WardId_data[variable].std()
            }
    
    # Store WardId multivariate results
    for WardId in df['WardId'].unique():
        WardId_data = df[df['WardId'] == WardId]
        multivar_outliers = WardId_data[WardId_data['multivar_outlier']]
        results['WardId']['multivariate'][WardId] = {
            'total_observations': len(WardId_data),
            'outliers_count': len(multivar_outliers),
            'outliers_percentage': (len(multivar_outliers)/len(WardId_data)*100),
            'outliers': multivar_outliers
        }
    
    # Print WardId level statistics
    for WardId in df['WardId'].unique():
        print(f"\nWardId: {WardId}")
        print("-" * 50)
        
        # WardId univariate statistics
        print("Univariate Analysis:")
        print(f"{'Variable':<15} {'Mean':>10} {'Median':>10} {'Std Dev':>10} {'Outliers':>10} {'Total':>10}")
        print("-" * 70)
        
        for variable in variables:
            stats = results['WardId']['univariate'][variable]['WardId_stats'][WardId]
            print(f"{variable:<15} {stats['mean']:10.2f} {stats['median']:10.2f} "
                  f"{stats['std_dev']:10.2f} {stats['outliers_count']:10d} {stats['total_observations']:10d}")
        
        # WardId multivariate statistics
        print("\nMultivariate Analysis:")
        multivar_stats = results['WardId']['multivariate'][WardId]
        print(f"Total observations: {multivar_stats['total_observations']}")
        print(f"Multivariate outliers: {multivar_stats['outliers_count']}")
        print(f"Percentage multivariate outliers: {multivar_stats['outliers_percentage']:.1f}%")
        
        # Print WardId level outlier details
        print("\nDetailed Outlier Information:")
        
        print("\nUnivariate Outliers:")
        for variable in variables:
            WardId_outliers = df[(df['WardId'] == WardId) & (df['is_outlier'])]
            if not WardId_outliers.empty:
                print(f"\nOutliers for {variable}:")
                print(WardId_outliers[['DataCollectionExerciseId', 'HouseHoldId', 
                                   variable, 'z_score']].to_string())
        
        print("\nMultivariate Outliers:")
        WardId_multivar_outliers = df[(df['WardId'] == WardId) & (df['multivar_outlier'])]
        if not WardId_multivar_outliers.empty:
            print(WardId_multivar_outliers[['DataCollectionExerciseId', 'HouseHoldId'] + 
                                      variables + ['mahalanobis_dist']].to_string())
    
    return df, results

def detect_multivariate_outliers_by_WardId(df, variables, WardId_column='WardId', tolerance=1e-20):
    """
    Detect multivariate outliers, either at CountyId level (WardId_column=None) or WardId level
    """
    df['mahalanobis_dist'] = np.nan
    df['multivar_outlier'] = False
    
    if WardId_column is None:
        # CountyId level analysis
        try:
            X = df[variables].values
            mean_vec = np.mean(X, axis=0)
            cov_matrix = np.cov(X, rowvar=False)
            inv_covmat = np.linalg.pinv(cov_matrix, rcond=tolerance)
            
            for idx in range(len(X)):
                diff = X[idx] - mean_vec
                df.iloc[idx, df.columns.get_loc('mahalanobis_dist')] = \
                    np.sqrt(diff.dot(inv_covmat).dot(diff))
            
            threshold = chi2.ppf(0.95, df=len(variables))
            df['multivar_outlier'] = df['mahalanobis_dist'] > threshold
            
        except Exception as e:
            print(f"Error in CountyId-level multivariate analysis: {str(e)}")
        
        return df
    
    # WardId level analysis
    for WardId in df[WardId_column].unique():
        WardId_data = df[df[WardId_column] == WardId].copy()
        
        if len(WardId_data) <= len(variables):
            print(f"Warning: WardId {WardId} has insufficient data for multivariate analysis")
            continue
            
        try:
            X = WardId_data[variables].values
            mean_vec = np.mean(X, axis=0)
            cov_matrix = np.cov(X, rowvar=False)
            inv_covmat = np.linalg.pinv(cov_matrix, rcond=tolerance)
            
            for idx in range(len(X)):
                diff = X[idx] - mean_vec
                WardId_data.iloc[idx, WardId_data.columns.get_loc('mahalanobis_dist')] = \
                    np.sqrt(diff.dot(inv_covmat).dot(diff))
            
            threshold = chi2.ppf(0.95, df=len(variables))
            WardId_data['multivar_outlier'] = WardId_data['mahalanobis_dist'] > threshold
            
            df.loc[df[WardId_column] == WardId, 'mahalanobis_dist'] = WardId_data['mahalanobis_dist']
            df.loc[df[WardId_column] == WardId, 'multivar_outlier'] = WardId_data['multivar_outlier']
            
        except Exception as e:
            print(f"Error in multivariate analysis for WardId {WardId}: {str(e)}")
            continue
    
    return df

def detect_univariate_outliers_by_WardId(df, variable, z_threshold=2.5, WardId_column='WardId'):
    """
    Detect univariate outliers, either at CountyId level (WardId_column=None) or WardId level
    """
    df['z_score'] = np.nan
    df['is_outlier'] = False
    
    if WardId_column is None:
        # CountyId level analysis
        try:
            df['z_score'] = np.abs(stats.zscore(df[variable]))
            df['is_outlier'] = df['z_score'] > z_threshold
        except Exception as e:
            print(f"Error in CountyId-level univariate analysis: {str(e)}")
        
        return df
    
    # WardId level analysis
    for WardId in df[WardId_column].unique():
        WardId_data = df[df[WardId_column] == WardId].copy()
        
        if len(WardId_data) <= 2:
            print(f"Warning: WardId {WardId} has insufficient data for univariate analysis")
            continue
            
        try:
            WardId_data['z_score'] = np.abs(stats.zscore(WardId_data[variable]))
            WardId_data['is_outlier'] = WardId_data['z_score'] > z_threshold
            
            df.loc[df[WardId_column] == WardId, 'z_score'] = WardId_data['z_score']
            df.loc[df[WardId_column] == WardId, 'is_outlier'] = WardId_data['is_outlier']
            
        except Exception as e:
            print(f"Error in univariate analysis for WardId {WardId}: {str(e)}")
            continue
    
    return df

# Run the analysis
#df, results = analyze_muac_data(df)

import json
from datetime import datetime
import numpy as np
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Index):
            return list(obj)
        return super().default(obj)
    
def convert_crop_production_results_to_json(results):
    """
    Convert crop production outlier detection results to JSON-serializable format
    """
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):  # Add this line
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_results = {
        'analysis_type': 'crop_production',
        'dataset_info': convert_numpy_types(results['dataset_info']),
        'column_stats': convert_numpy_types(results['column_stats']),
        'univariate_analysis': {},
        'multivariate_analysis': {},
        'ward_level_analysis': {}
    }

    # Convert univariate results
    for column, crop_data in results['univariate'].items():
        json_results['univariate_analysis'][str(column)] = {}
        for crop, data in crop_data.items():
            json_results['univariate_analysis'][str(column)][str(crop)] = {
                'total_observations': int(data['total_observations']),
                'outliers_count': int(data['outliers_count']),
                'outliers_percentage': float(data['outliers_percentage']),
                'mean': float(data['mean']),
                'std': float(data['std']),
                'outliers': convert_numpy_types(data['outliers'])
            }

    # Convert multivariate results
    if 'error' in results['multivariate']:
        json_results['multivariate_analysis'] = {
            'status': 'error',
            'message': str(results['multivariate']['error'])
        }
    else:
        json_results['multivariate_analysis'] = {
            'total_observations': int(results['multivariate']['total_observations']),
            'outliers_count': int(results['multivariate']['outliers_count']),
            'outliers_percentage': float(results['multivariate']['outliers_percentage']),
            'mahalanobis_threshold': float(results['multivariate']['mahalanobis_threshold']),
            'variables_used': results['multivariate']['variables_used'],
            'outliers': convert_numpy_types(results['multivariate']['outliers'])
        }

    # Convert ward-level results
    for ward, ward_data in results['ward_level'].items():
        json_results['ward_level_analysis'][str(ward)] = {
            'univariate': {}
        }
        for column, crop_data in ward_data['univariate'].items():
            json_results['ward_level_analysis'][str(ward)]['univariate'][str(column)] = {}
            for crop, data in crop_data.items():
                json_results['ward_level_analysis'][str(ward)]['univariate'][str(column)][str(crop)] = {
                    'total_observations': int(data['total_observations']),
                    'outliers_count': int(data['outliers_count']),
                    'outliers': convert_numpy_types(data['outliers'])
                }

    return json_results

def convert_livestock_production_results_to_json(results):
    """
    Convert livestock production outlier detection results to JSON-serializable format
    """
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):  # Add this line
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_results = {
        'analysis_type': 'livestock_production',
        'dataset_info': convert_numpy_types(results['dataset_info']),
        'column_stats': convert_numpy_types(results['column_stats']),
        'univariate_analysis': {},
        'multivariate_analysis': {},
        'ward_level_analysis': {}
    }

    # Convert univariate results
    for column, animal_data in results['univariate'].items():
        json_results['univariate_analysis'][str(column)] = {}
        for animal, data in animal_data.items():
            json_results['univariate_analysis'][str(column)][str(animal)] = {
                'total_observations': int(data['total_observations']),
                'outliers_count': int(data['outliers_count']),
                'outliers_percentage': float(data['outliers_percentage']),
                'mean': float(data['mean']),
                'std': float(data['std']),
                'outliers': convert_numpy_types(data['outliers'])
            }

    # Convert multivariate results
    if 'error' in results['multivariate']:
        json_results['multivariate_analysis'] = {
            'status': 'error',
            'message': str(results['multivariate']['error'])
        }
    else:
        json_results['multivariate_analysis'] = {
            'total_observations': int(results['multivariate']['total_observations']),
            'outliers_count': int(results['multivariate']['outliers_count']),
            'outliers_percentage': float(results['multivariate']['outliers_percentage']),
            'mahalanobis_threshold': float(results['multivariate']['mahalanobis_threshold']),
            'variables_used': results['multivariate']['variables_used'],
            'outliers': convert_numpy_types(results['multivariate']['outliers'])
        }

    # Convert ward-level results
    for ward, ward_data in results['ward_level'].items():
        json_results['ward_level_analysis'][str(ward)] = {
            'univariate': {}
        }
        for column, animal_data in ward_data['univariate'].items():
            json_results['ward_level_analysis'][str(ward)]['univariate'][str(column)] = {}
            for animal, data in animal_data.items():
                json_results['ward_level_analysis'][str(ward)]['univariate'][str(column)][str(animal)] = {
                    'total_observations': int(data['total_observations']),
                    'outliers_count': int(data['outliers_count']),
                    'outliers': convert_numpy_types(data['outliers'])
                }

    return json_results

def convert_coping_strategies_results_to_json(results):
    """Convert coping strategies outlier detection results to JSON-serializable format"""
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):  # Add this line
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_results = {
        'analysis_type': 'coping_strategies',
        'dataset_info': convert_numpy_types(results['dataset_info']),
        'column_stats': convert_numpy_types(results['column_stats']),
        'univariate_analysis': {},
        'multivariate_analysis': {},
        'ward_level_analysis': {}
    }

    # Convert univariate results
    for column, strategy_data in results['univariate'].items():
        json_results['univariate_analysis'][str(column)] = {}
        for strategy, data in strategy_data.items():
            json_results['univariate_analysis'][str(column)][str(strategy)] = {
                'total_observations': int(data['total_observations']),
                'outliers_count': int(data['outliers_count']),
                'outliers_percentage': float(data['outliers_percentage']),
                'mean': float(data['mean']),
                'std': float(data['std']),
                'outliers': convert_numpy_types(data['outliers'])
            }

    # Convert multivariate results (similar structure to previous)
    if 'error' in results['multivariate']:
        json_results['multivariate_analysis'] = {
            'status': 'error',
            'message': str(results['multivariate']['error'])
        }
    else:
        json_results['multivariate_analysis'] = convert_numpy_types(results['multivariate'])

    # Convert ward-level results
    for ward, ward_data in results['ward_level'].items():
        json_results['ward_level_analysis'][str(ward)] = convert_numpy_types(ward_data)

    return json_results

def convert_food_consumption_results_to_json(results):
    """Convert food consumption outlier detection results to JSON-serializable format"""
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):  # Add this line
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_results = {
        'analysis_type': 'food_consumption',
        'dataset_info': convert_numpy_types(results['dataset_info']),
        'column_stats': convert_numpy_types(results['column_stats']),
        'univariate_analysis': {},
        'multivariate_analysis': {},
        'ward_level_analysis': {}
    }

    # Convert univariate results
    for column, food_data in results['univariate'].items():
        json_results['univariate_analysis'][str(column)] = {}
        for food_type, data in food_data.items():
            json_results['univariate_analysis'][str(column)][str(food_type)] = {
                'total_observations': int(data['total_observations']),
                'outliers_count': int(data['outliers_count']),
                'outliers_percentage': float(data['outliers_percentage']),
                'mean': float(data['mean']),
                'std': float(data['std']),
                'outliers': convert_numpy_types(data['outliers'])
            }

    # Convert multivariate and ward-level results (similar structure)
    if 'error' in results['multivariate']:
        json_results['multivariate_analysis'] = {
            'status': 'error',
            'message': str(results['multivariate']['error'])
        }
    else:
        json_results['multivariate_analysis'] = convert_numpy_types(results['multivariate'])

    for ward, ward_data in results['ward_level'].items():
        json_results['ward_level_analysis'][str(ward)] = convert_numpy_types(ward_data)

    return json_results

def convert_muac_results_to_json(results):
    """Convert MUAC analysis results to JSON-serializable format"""
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):  # Add this line
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_results = {
        'analysis_type': 'muac',
        'county_level': {
            'univariate': convert_numpy_types(results['CountyId']['univariate']),
            'multivariate': convert_numpy_types(results['CountyId']['multivariate'])
        },
        'ward_level': {
            'univariate': convert_numpy_types(results['WardId']['univariate']),
            'multivariate': convert_numpy_types(results['WardId']['multivariate'])
        }
    }

    return json_results

def convert_milk_production_results_to_json(results):
    """Convert milk production outlier detection results to JSON-serializable format"""
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):  # Add this line
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_results = {
        'analysis_type': 'milk_production',
        'dataset_info': convert_numpy_types(results['dataset_info']),
        'column_stats': convert_numpy_types(results['column_stats']),
        'univariate_analysis': {},
        'multivariate_analysis': {},
        'ward_level_analysis': {}
    }

    # Convert univariate results
    for column, animal_data in results['univariate'].items():
        json_results['univariate_analysis'][str(column)] = {}
        for animal, data in animal_data.items():
            json_results['univariate_analysis'][str(column)][str(animal)] = {
                'total_observations': int(data['total_observations']),
                'outliers_count': int(data['outliers_count']),
                'outliers_percentage': float(data['outliers_percentage']),
                'mean': float(data['mean']),
                'std': float(data['std']),
                'outliers': convert_numpy_types(data['outliers'])
            }

    # Convert multivariate results
    if 'error' in results['multivariate']:
        json_results['multivariate_analysis'] = {
            'status': 'error',
            'message': str(results['multivariate']['error'])
        }
    else:
        json_results['multivariate_analysis'] = convert_numpy_types(results['multivariate'])

    # Convert ward-level results
    for ward, ward_data in results['ward_level'].items():
        json_results['ward_level_analysis'][str(ward)] = convert_numpy_types(ward_data)

    return json_results

import json
from datetime import datetime

def combine_all_outlier_analyses(crop_results=None, livestock_results=None, 
                               milk_results=None, coping_results=None, 
                               food_results=None, muac_results=None):
    """
    Combine all outlier analyses into a single JSON structure
    Ordered logically with milk production following livestock production
    """
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'analyses': {},
        'summary': {
            'total_records_analyzed': {},
            'unique_wards': {},
            'total_outliers': {}
        }
    }

    # Add crop production analysis
    if crop_results:
        combined_results['analyses']['crop_production'] = convert_crop_production_results_to_json(crop_results)
        combined_results['summary']['total_records_analyzed']['crop_production'] = crop_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['crop_production'] = crop_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['crop_production'] = {
            'univariate': sum(data['outliers_count'] for col_data in crop_results['univariate'].values() for data in col_data.values()),
            'multivariate': crop_results['multivariate'].get('outliers_count', 0) if isinstance(crop_results['multivariate'], dict) else 0
        }

    # Add livestock production analysis
    if livestock_results:
        combined_results['analyses']['livestock_production'] = convert_livestock_production_results_to_json(livestock_results)
        combined_results['summary']['total_records_analyzed']['livestock_production'] = livestock_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['livestock_production'] = livestock_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['livestock_production'] = {
            'univariate': sum(data['outliers_count'] for col_data in livestock_results['univariate'].values() for data in col_data.values()),
            'multivariate': livestock_results['multivariate'].get('outliers_count', 0) if isinstance(livestock_results['multivariate'], dict) else 0
        }

    # Add milk production analysis
    if milk_results:
        combined_results['analyses']['milk_production'] = convert_milk_production_results_to_json(milk_results)
        combined_results['summary']['total_records_analyzed']['milk_production'] = milk_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['milk_production'] = milk_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['milk_production'] = {
            'univariate': sum(data['outliers_count'] for col_data in milk_results['univariate'].values() for data in col_data.values()),
            'multivariate': milk_results['multivariate'].get('outliers_count', 0) if isinstance(milk_results['multivariate'], dict) else 0
        }

    # Add coping strategies analysis
    if coping_results:
        combined_results['analyses']['coping_strategies'] = convert_coping_strategies_results_to_json(coping_results)
        combined_results['summary']['total_records_analyzed']['coping_strategies'] = coping_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['coping_strategies'] = coping_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['coping_strategies'] = {
            'univariate': sum(data['outliers_count'] for col_data in coping_results['univariate'].values() for data in col_data.values()),
            'multivariate': coping_results['multivariate'].get('outliers_count', 0) if isinstance(coping_results['multivariate'], dict) else 0
        }

    # Add food consumption analysis
    if food_results:
        combined_results['analyses']['food_consumption'] = convert_food_consumption_results_to_json(food_results)
        combined_results['summary']['total_records_analyzed']['food_consumption'] = food_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['food_consumption'] = food_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['food_consumption'] = {
            'univariate': sum(data['outliers_count'] for col_data in food_results['univariate'].values() for data in col_data.values()),
            'multivariate': food_results['multivariate'].get('outliers_count', 0) if isinstance(food_results['multivariate'], dict) else 0
        }

    # Add MUAC analysis
    if muac_results:
        combined_results['analyses']['muac'] = convert_muac_results_to_json(muac_results)
        combined_results['summary']['total_outliers']['muac'] = {
            'county_level': {
                'univariate': sum(data['outliers_count'] for data in muac_results['CountyId']['univariate'].values()),
                'multivariate': muac_results['CountyId']['multivariate']['outliers_count']
            },
            'ward_level': {
                'univariate': sum(sum(ward_data['outliers_count'] for ward_data in var_data['WardId_stats'].values()) 
                                for var_data in muac_results['WardId']['univariate'].values()),
                'multivariate': sum(ward_data['outliers_count'] for ward_data in muac_results['WardId']['multivariate'].values())
            }
        }

    return combined_results

def combine_all_outlier_analyses(crop_results=None, livestock_results=None, 
                               milk_results=None, coping_results=None, 
                               food_results=None, muac_results=None):
    """
    Combine all outlier analyses into a single JSON structure
    Ordered logically with milk production following livestock production
    """
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'analyses': {},
        'summary': {
            'total_records_analyzed': {},
            'unique_wards': {},
            'total_outliers': {}
        }
    }

    # Add crop production analysis
    if crop_results:
        combined_results['analyses']['crop_production'] = convert_crop_production_results_to_json(crop_results)
        combined_results['summary']['total_records_analyzed']['crop_production'] = crop_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['crop_production'] = crop_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['crop_production'] = {
            'univariate': sum(data['outliers_count'] for col_data in crop_results['univariate'].values() for data in col_data.values()),
            'multivariate': crop_results['multivariate'].get('outliers_count', 0) if isinstance(crop_results['multivariate'], dict) else 0
        }

    # Add livestock production analysis
    if livestock_results:
        combined_results['analyses']['livestock_production'] = convert_livestock_production_results_to_json(livestock_results)
        combined_results['summary']['total_records_analyzed']['livestock_production'] = livestock_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['livestock_production'] = livestock_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['livestock_production'] = {
            'univariate': sum(data['outliers_count'] for col_data in livestock_results['univariate'].values() for data in col_data.values()),
            'multivariate': livestock_results['multivariate'].get('outliers_count', 0) if isinstance(livestock_results['multivariate'], dict) else 0
        }

    # Add milk production analysis
    if milk_results:
        combined_results['analyses']['milk_production'] = convert_milk_production_results_to_json(milk_results)
        combined_results['summary']['total_records_analyzed']['milk_production'] = milk_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['milk_production'] = milk_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['milk_production'] = {
            'univariate': sum(data['outliers_count'] for col_data in milk_results['univariate'].values() for data in col_data.values()),
            'multivariate': milk_results['multivariate'].get('outliers_count', 0) if isinstance(milk_results['multivariate'], dict) else 0
        }

    # Add coping strategies analysis
    if coping_results:
        combined_results['analyses']['coping_strategies'] = convert_coping_strategies_results_to_json(coping_results)
        combined_results['summary']['total_records_analyzed']['coping_strategies'] = coping_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['coping_strategies'] = coping_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['coping_strategies'] = {
            'univariate': sum(data['outliers_count'] for col_data in coping_results['univariate'].values() for data in col_data.values()),
            'multivariate': coping_results['multivariate'].get('outliers_count', 0) if isinstance(coping_results['multivariate'], dict) else 0
        }

    # Add food consumption analysis
    if food_results:
        combined_results['analyses']['food_consumption'] = convert_food_consumption_results_to_json(food_results)
        combined_results['summary']['total_records_analyzed']['food_consumption'] = food_results['dataset_info']['total_records']
        combined_results['summary']['unique_wards']['food_consumption'] = food_results['dataset_info']['unique_wards']
        combined_results['summary']['total_outliers']['food_consumption'] = {
            'univariate': sum(data['outliers_count'] for col_data in food_results['univariate'].values() for data in col_data.values()),
            'multivariate': food_results['multivariate'].get('outliers_count', 0) if isinstance(food_results['multivariate'], dict) else 0
        }

    # Add MUAC analysis
    if muac_results:
        combined_results['analyses']['muac'] = convert_muac_results_to_json(muac_results)
        combined_results['summary']['total_outliers']['muac'] = {
            'county_level': {
                'univariate': sum(data['outliers_count'] for data in muac_results['CountyId']['univariate'].values()),
                'multivariate': muac_results['CountyId']['multivariate']['outliers_count']
            },
            'ward_level': {
                'univariate': sum(sum(ward_data['outliers_count'] for ward_data in var_data['WardId_stats'].values()) 
                                for var_data in muac_results['WardId']['univariate'].values()),
                'multivariate': sum(ward_data['outliers_count'] for ward_data in muac_results['WardId']['multivariate'].values())
            }
        }

    return combined_results


# Run your analyses
crop_results = detect_outliers_crop_production(
    crop_df=crop_df,
    id_column='CropId',
    value_columns=['AcresPlantedInLastFourWks', 'AcresHarvestedInLastFourWks', 
                   'KgsHarvestedInLastFourWks', 'OwnProductionStockInKg', 
                   'KgsSoldInLastFourWks', 'PricePerKg'],
    z_threshold=2.5
)

livestock_results = detect_outliers_livestock_production(
    livestock_df=livestock_df,
    id_column='AnimalId',
    value_columns=['NumberKeptToday', 'NumberBornInLastFourWeeks', 
                   'NumberPurchasedInLastFourWeeks', 'NumberSoldInLastFourWeeks', 
                   'AveragePricePerAnimalSold', 'NumberDiedDuringLastFourWeeks'],
    z_threshold=2.5
)

milk_results = detect_outliers_milk(
    milk_df=milk_df,
    id_column='AnimalId',
    value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 
                   'DailyQntySoldInLtrs', 'PricePerLtr'],
    z_threshold=2.5
)

coping_results = detect_outliers_Copying_Strategies(
    coping_df=coping_df,
    id_column='CopyingStrategyId',
    value_columns=['NumOfCopingDays'],
    z_threshold=2.5
)

food_results = detect_outliers_Food_Consumption(
    food_c_df=food_c_df,
    id_column='FoodTypeId',
    value_columns=['NumDaysEaten'],
    z_threshold=2.5
)

df, muac_results = analyze_muac_data(
    df=df,
    variables=['AgeInMonths', 'MuacInMillimeters'],
    z_threshold=2.5
)

# Combine all results
combined_json = combine_all_outlier_analyses(
    crop_results=crop_results,
    livestock_results=livestock_results,
    milk_results=milk_results,
    coping_results=coping_results,
    food_results=food_results,
    muac_results=muac_results
)

# Save to file using the custom encoder
with open('combined_outlier_analysis.json', 'w') as f:
    json.dump(combined_json, f, indent=2, cls=CustomJSONEncoder)
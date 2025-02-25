import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2

def detect_outliers_crop_production(crop_df, id_column='CropId', 
                               value_columns=['AcresPlantedInLastFourWks', 'AcresHarvestedInLastFourWks', 
                                             'KgsHarvestedInLastFourWks', 'OwnProductionStockInKg', 
                                             'KgsSoldInLastFourWks', 'PricePerKg'],
                               z_threshold=2.5):
    """Detects outliers in crop production data."""
    results = {'dataset_info': {}, 'column_stats': {}, 'univariate': {}, 'multivariate': {}}

    # Dataset summary
    results['dataset_info'] = {
        'total_records': len(crop_df),
        'unique_wards': crop_df['WardId'].nunique(),
        'unique_crops': crop_df[id_column].nunique()
    }

    # Check stats for each column
    for column in value_columns:
        if column in crop_df.columns:
            results['column_stats'][column] = {
                'mean': crop_df[column].mean(),
                'median': crop_df[column].median(),
                'std': crop_df[column].std(),
                'min': crop_df[column].min(),
                'max': crop_df[column].max()
            }

    # Univariate Outlier Detection
    for column in value_columns:
        if column in crop_df.columns:
            crop_df['z_score'] = np.abs(stats.zscore(crop_df[column]))
            outliers = crop_df[crop_df['z_score'] > z_threshold]
            results['univariate'][column] = {
                'total_observations': len(crop_df),
                'outliers_count': len(outliers),
                'outliers': outliers.to_dict(orient='records')
            }

    return results

def detect_outliers_livestock_production(livestock_df, id_column='AnimalId', 
                               value_columns=['NumberKeptToday', 'NumberBornInLastFourWeeks', 
                                             'NumberPurchasedInLastFourWeeks', 'NumberSoldInLastFourWeeks', 
                                             'AveragePricePerAnimalSold', 'NumberDiedDuringLastFourWeeks'],
                               z_threshold=2.5):
    """Detects outliers in livestock production data."""
    results = {'dataset_info': {}, 'column_stats': {}, 'univariate': {}}

    # Dataset summary
    results['dataset_info'] = {
        'total_records': len(livestock_df),
        'unique_wards': livestock_df['WardId'].nunique(),
        'unique_animals': livestock_df[id_column].nunique()
    }

    # Univariate Outlier Detection
    for column in value_columns:
        if column in livestock_df.columns:
            livestock_df['z_score'] = np.abs(stats.zscore(livestock_df[column]))
            outliers = livestock_df[livestock_df['z_score'] > z_threshold]
            results['univariate'][column] = {
                'total_observations': len(livestock_df),
                'outliers_count': len(outliers),
                'outliers': outliers.to_dict(orient='records')
            }

    return results

def detect_outliers_milk(milk_df, id_column='AnimalId', 
                               value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 
                                             'DailyQntySoldInLtrs', 'PricePerLtr'],
                               z_threshold=2.5):
    """Detects outliers in milk production data."""
    results = {'dataset_info': {}, 'column_stats': {}, 'univariate': {}}

    for column in value_columns:
        if column in milk_df.columns:
            milk_df['z_score'] = np.abs(stats.zscore(milk_df[column]))
            outliers = milk_df[milk_df['z_score'] > z_threshold]
            results['univariate'][column] = {
                'total_observations': len(milk_df),
                'outliers_count': len(outliers),
                'outliers': outliers.to_dict(orient='records')
            }

    return results

def detect_outliers_Copying_Strategies(coping_df, id_column='CopyingStrategyId', 
                               value_columns=['NumOfCopingDays'],
                               z_threshold=2.5):
    """Detects outliers in coping strategies data."""
    results = {'dataset_info': {}, 'column_stats': {}, 'univariate': {}}

    for column in value_columns:
        if column in coping_df.columns:
            coping_df['z_score'] = np.abs(stats.zscore(coping_df[column]))
            outliers = coping_df[coping_df['z_score'] > z_threshold]
            results['univariate'][column] = {
                'total_observations': len(coping_df),
                'outliers_count': len(outliers),
                'outliers': outliers.to_dict(orient='records')
            }

    return results

def detect_outliers_Food_Consumption(food_c_df, id_column='FoodTypeId', 
                               value_columns=['NumDaysEaten'],
                               z_threshold=2.5):
    """Detects outliers in food consumption data."""
    results = {'dataset_info': {}, 'column_stats': {}, 'univariate': {}}

    for column in value_columns:
        if column in food_c_df.columns:
            food_c_df['z_score'] = np.abs(stats.zscore(food_c_df[column]))
            outliers = food_c_df[food_c_df['z_score'] > z_threshold]
            results['univariate'][column] = {
                'total_observations': len(food_c_df),
                'outliers_count': len(outliers),
                'outliers': outliers.to_dict(orient='records')
            }

    return results

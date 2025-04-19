#importing libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2



def detect_outliers(input_df, id_column='AnimalId', 
                            value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 'DailyQntySoldInLtrs', 'PricePerLtr'],
                            z_threshold=2.5):
    """
    Comprehensive outlier detection for Indicator data with multiple measurements
    Returns a DataFrame with outlier details including all requested additional columns
    
    Parameters:
    -----------
    input_df : pandas DataFrame
        Input data with required columns
    id_column : str
        Name of the Indicator ID column
    value_columns : list
        List of measurement column names to analyze
    z_threshold : float
        Z-score threshold for univariate outlier detection
    
    Returns:
    --------
    pandas DataFrame
        DataFrame containing outlier details with all requested columns
    """
    
    # Initialize empty lists to store all outlier records
    outlier_records = []
    
    # List of additional columns to include
    additional_columns = [
        'HhaQuestionnaireSessionId', 'CountyId', 'LivelihoodZoneId', 
        'WardId', 'HouseHoldId', 'SubCountyId', 'DataCollectionExerciseId',
        'ExerciseStartDate'
    ]
    
    # Verify which additional columns actually exist in the input dataframe
    existing_additional_cols = [col for col in additional_columns if col in input_df.columns]
    
    # Calculate population statistics for each column
    population_stats = {}
    for col in value_columns:
        if col in input_df.columns:
            population_stats[col] = {
                'mean': input_df[col].mean(),
                'std': input_df[col].std()
            }
    
    # 1. COUNTY-LEVEL UNIVARIATE OUTLIERS
    for value_column in value_columns:
        if value_column not in input_df.columns:
            continue
        
        # Get valid Indicators with variation
        valid_Indicators = []
        for Indicator in input_df[id_column].unique():
            Indicator_data = input_df[input_df[id_column] == Indicator][value_column].dropna()
            if len(Indicator_data) > 1 and Indicator_data.std() > 0:
                valid_Indicators.append(Indicator)
        
        # Detect univariate outliers for each valid Indicator
        for Indicator in valid_Indicators:
            Indicator_data = input_df[input_df[id_column] == Indicator].copy()
            Indicator_data = Indicator_data.dropna(subset=[value_column])
            
            if len(Indicator_data) <= 1:
                continue
                
            # Calculate reference stats (within this Indicator)
            reference_mean = Indicator_data[value_column].mean()
            reference_std = Indicator_data[value_column].std()
            
            # Calculate z-scores
            Indicator_data['z_score'] = np.abs(stats.zscore(Indicator_data[value_column]))
            outliers = Indicator_data[Indicator_data['z_score'] > z_threshold]
            
            # Add each outlier to our records
            for _, row in outliers.iterrows():
                record = {
                    'OutlierValue': row[value_column],
                    'level': 'county',
                    'outlier_type': 'univariate',
                    'entity_id': Indicator,
                    'test_statistic_value': row['z_score'],
                    'test_type': 'Z-score',
                    'reference_mean': reference_mean,
                    'reference_std': reference_std,
                    'population_mean': population_stats[value_column]['mean'],
                    'population_std': population_stats[value_column]['std'],
                    'analyzed_column': value_column
                }
                
                # Add additional columns if they exist in the original data
                for col in existing_additional_cols:
                    record[col] = row[col]
                
                outlier_records.append(record)
    
    # 2. COUNTY-LEVEL MULTIVARIATE OUTLIERS
    available_columns = [col for col in value_columns if col in input_df.columns]
    
    if len(available_columns) >= 2:
        try:
            # Create a copy for multivariate analysis
            mv_input_df = input_df.copy()
            
            # Impute missing values with median for each Indicator
            for Indicator in input_df[id_column].unique():
                Indicator_mask = mv_input_df[id_column] == Indicator
                for col in available_columns:
                    median_val = mv_input_df.loc[Indicator_mask, col].median()
                    if pd.notna(median_val):
                        mv_input_df.loc[Indicator_mask, col] = mv_input_df.loc[Indicator_mask, col].fillna(median_val)
            
            # Drop any remaining rows with NaN after imputation
            mv_input_df = mv_input_df.dropna(subset=available_columns)
            
            if len(mv_input_df) >= 10:
                # Calculate Mahalanobis distance
                X = mv_input_df[available_columns].values
                mean_vec = np.mean(X, axis=0)
                cov_matrix = np.cov(X, rowvar=False)
                
                try:
                    inv_covmat = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    inv_covmat = np.linalg.pinv(cov_matrix)
                
                mv_input_df['mahalanobis_dist'] = np.zeros(len(mv_input_df))
                for idx in range(len(X)):
                    diff = X[idx] - mean_vec
                    mv_input_df.loc[mv_input_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))
                
                # Chi-square threshold for multivariate outliers
                threshold = chi2.ppf(0.975, df=len(available_columns))
                mv_input_df['multivar_outlier'] = mv_input_df['mahalanobis_dist'] > threshold
                
                # Calculate reference stats (global for multivariate)
                reference_mean = {col: mv_input_df[col].mean() for col in available_columns}
                reference_std = {col: mv_input_df[col].std() for col in available_columns}
                
                # Add multivariate outliers to our records
                for _, row in mv_input_df[mv_input_df['multivar_outlier']].iterrows():
                    record = {
                        'OutlierValue': '|'.join([str(row[col]) for col in available_columns]),
                        'level': 'county',
                        'outlier_type': 'multivariate',
                        'entity_id': row[id_column],
                        'test_statistic_value': row['mahalanobis_dist'],
                        'test_type': 'Mahalanobis distance',
                        'reference_mean': '|'.join([str(reference_mean[col]) for col in available_columns]),
                        'reference_std': '|'.join([str(reference_std[col]) for col in available_columns]),
                        'population_mean': '|'.join([str(population_stats[col]['mean']) for col in available_columns]),
                        'population_std': '|'.join([str(population_stats[col]['std']) for col in available_columns]),
                        'analyzed_column': '|'.join(available_columns)
                    }
                    
                    # Add additional columns if they exist in the original data
                    for col in existing_additional_cols:
                        record[col] = row[col]
                    
                    outlier_records.append(record)
        
        except Exception as e:
            print(f"Multivariate analysis error: {str(e)}")
    
    # 3. WARD-LEVEL UNIVARIATE OUTLIERS
    if 'WardId' in input_df.columns:
        for ward in input_df['WardId'].unique():
            ward_data = input_df[input_df['WardId'] == ward]
            
            for value_column in value_columns:
                if value_column not in input_df.columns:
                    continue
                    
                for Indicator in input_df[id_column].unique():
                    Indicator_ward_data = ward_data[ward_data[id_column] == Indicator].copy()
                    Indicator_ward_data = Indicator_ward_data.dropna(subset=[value_column])
                    
                    if len(Indicator_ward_data) > 1 and Indicator_ward_data[value_column].std() > 0:
                        # Calculate reference stats (within this ward and Indicator)
                        reference_mean = Indicator_ward_data[value_column].mean()
                        reference_std = Indicator_ward_data[value_column].std()
                        
                        # Calculate z-scores
                        Indicator_ward_data['z_score'] = np.abs(stats.zscore(Indicator_ward_data[value_column]))
                        ward_outliers = Indicator_ward_data[Indicator_ward_data['z_score'] > z_threshold]
                        
                        # Add ward-level outliers to our records
                        for _, row in ward_outliers.iterrows():
                            record = {
                                'OutlierValue': row[value_column],
                                'level': 'ward',
                                'outlier_type': 'univariate',
                                'entity_id': Indicator,
                                'test_statistic_value': row['z_score'],
                                'test_type': 'Z-score',
                                'reference_mean': reference_mean,
                                'reference_std': reference_std,
                                'population_mean': population_stats[value_column]['mean'],
                                'population_std': population_stats[value_column]['std'],
                                'analyzed_column': value_column
                            }
                            
                            # Add additional columns if they exist in the original data
                            for col in existing_additional_cols:
                                record[col] = row[col]
                            
                            outlier_records.append(record)
    
    # Convert the records to a DataFrame
    if outlier_records:
        outlier_df = pd.DataFrame(outlier_records)
        
        # Define the column order
        base_columns = [
            'OutlierValue', 'level', 'outlier_type', 'entity_id',
            'test_statistic_value', 'test_type', 'reference_mean',
            'reference_std', 'population_mean', 'population_std',
            'analyzed_column'
        ]
        
        # Final column order (base columns + additional columns)
        final_columns = base_columns + existing_additional_cols
        
        return outlier_df[final_columns].sort_values(
            by=['level', 'outlier_type', 'entity_id', 'analyzed_column']
        )
    else:
        # Return empty DataFrame with all requested columns
        empty_columns = [
            'OutlierValue', 'level', 'outlier_type', 'entity_id',
            'test_statistic_value', 'test_type', 'reference_mean',
            'reference_std', 'population_mean', 'population_std',
            'analyzed_column'
        ] + existing_additional_cols
        
        return pd.DataFrame(columns=empty_columns)


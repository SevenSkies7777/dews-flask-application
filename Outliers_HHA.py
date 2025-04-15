#importing libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2

class HHA_Outliers:
    """Class returns Milk production Outliers."""


    def detect_outliers_multicolumn(self, df=None, id_column='AnimalId', 
                               value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 
                                             'DailyQntySoldInLtrs', 'PricePerLtr'],
                               z_threshold=2.5,
                               verbose=False):
        
        # Use class df if none provided
        if df is None:
            df = self.df
            
        # Check if dataframe is empty
        if df.empty:
            return {
                'dataset_info': {
                    'total_records': 0,
                    'unique_wards': 0,
                    'unique_entities': 0,
                    'unique_households': 0
                },
                'column_stats': {},
                'outliers': pd.DataFrame(),
                'error': 'Input dataframe is empty'
            }

        results = {
            'dataset_info': {},
            'column_stats': {},
            'outliers': pd.DataFrame()
        }

        # 1. Dataset Overview
        results['dataset_info'] = {
            'total_records': len(df),
            'unique_wards': df['WardId'].nunique() if 'WardId' in df.columns else 0,
            'unique_entities': df[id_column].nunique(),
            'unique_households': df['HouseHoldId'].nunique() if 'HouseHoldId' in df.columns else 0
        }
        
        # 2. Check stats for each column
        for column in value_columns:
            # Skip if column doesn't exist in dataframe
            if column not in df.columns:
                continue
                
            column_stats = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'null_count': df[column].isnull().sum(),
                'zero_count': (df[column] == 0).sum()
            }
            results['column_stats'][column] = column_stats
        
        # Lists to store all types of outliers
        all_outliers = []
        
        # 3. Process each measurement column - univariate outliers
        for value_column in value_columns:
            if value_column not in df.columns:
                continue
            
            zero_var_entities = []
            valid_entities = []
            
            # Calculate county-level mean and std dev for this column
            county_mean = df[value_column].mean()
            county_std = df[value_column].std()
            
            # 3.1 Check for zero variance entities in this column
            for entity_id in df[id_column].unique():
                entity_data = df[df[id_column] == entity_id][value_column]
                # Skip if all values are null
                if entity_data.isnull().all():
                    continue
                    
                # Handling nulls by dropping them for variance check
                entity_data = entity_data.dropna()
                
                if len(entity_data) == 0 or entity_data.std() == 0:
                    zero_var_entities.append(entity_id)
                else:
                    valid_entities.append(entity_id)
            
            # 3.2 Ward-level and County-level Univariate Outlier Detection for this column
            for entity_id in valid_entities:

                entity_data = df[df[id_column] == entity_id].copy()

                entity_data = entity_data.dropna(subset=[value_column])
                
                if len(entity_data) <= 1:
                    continue

                entity_mean = entity_data[value_column].mean()
                entity_std = entity_data[value_column].std()
                
                entity_data['z_score'] = np.abs(stats.zscore(entity_data[value_column]))

                county_outliers = entity_data[entity_data['z_score'] > z_threshold].copy()
                if len(county_outliers) > 0:

                    county_outliers['analyzed_column'] = value_column

                    county_outliers['OutlierValue'] = county_outliers[value_column]
                    
                    county_outliers['level'] = 'county'
                    county_outliers['outlier_type'] = 'univariate'
                    county_outliers['entity_id'] = entity_id
                    county_outliers['test_statistic_value'] = county_outliers['z_score']
                    county_outliers['test_type'] = 'Z-score'

                    county_outliers['reference_mean'] = entity_mean
                    county_outliers['reference_std'] = entity_std
                    county_outliers['population_mean'] = county_mean
                    county_outliers['population_std'] = county_std

                    all_outliers.append(county_outliers)

                if 'WardId' in df.columns:
                    for ward in df['WardId'].unique():
                        ward_data = entity_data[entity_data['WardId'] == ward].copy()

                        if len(ward_data) <= 1 or ward_data[value_column].isnull().all():
                            continue

                        ward_mean = df[df['WardId'] == ward][value_column].mean()
                        ward_std = df[df['WardId'] == ward][value_column].std()

                        entity_ward_mean = ward_data[value_column].mean()
                        entity_ward_std = ward_data[value_column].std()

                        if len(ward_data) > 1 and ward_data[value_column].std() > 0:
                            ward_data['z_score'] = np.abs(stats.zscore(ward_data[value_column]))
                            ward_outliers = ward_data[ward_data['z_score'] > z_threshold].copy()
                            
                            if len(ward_outliers) > 0:

                                ward_outliers['analyzed_column'] = value_column

                                ward_outliers['OutlierValue'] = ward_outliers[value_column]
                                
                                ward_outliers['level'] = 'ward'
                                ward_outliers['outlier_type'] = 'univariate'
                                ward_outliers['entity_id'] = entity_id
                                ward_outliers['test_statistic_value'] = ward_outliers['z_score']
                                ward_outliers['test_type'] = 'Z-score'

                                ward_outliers['reference_mean'] = entity_ward_mean
                                ward_outliers['reference_std'] = entity_ward_std
                                ward_outliers['population_mean'] = ward_mean
                                ward_outliers['population_std'] = ward_std

                                all_outliers.append(ward_outliers)
        
        # 4. Multivariate Outlier Detection 

        available_columns = [col for col in value_columns if col in df.columns]
        
        if len(available_columns) >= 2:  
            try:

                mv_df = df.copy()
                
                # Handling missing values For multivariate analysis: Strategy - Impute with median for each entity
                for entity_id in df[id_column].unique():
                    entity_mask = mv_df[id_column] == entity_id
                    for col in available_columns:
                        median_val = mv_df.loc[entity_mask, col].median()
                        # Only fill if median is not NaN
                        if pd.notna(median_val):
                            mv_df.loc[entity_mask, col] = mv_df.loc[entity_mask, col].fillna(median_val)
                
                mv_df = mv_df.dropna(subset=available_columns)
                
                if len(mv_df) >= 10:  

                    X = mv_df[available_columns].values
                    mean_vec = np.mean(X, axis=0)
                    cov_matrix = np.cov(X, rowvar=False)

                    county_means = {col: mv_df[col].mean() for col in available_columns}
                    county_stds = {col: mv_df[col].std() for col in available_columns}

                    try:
                        inv_covmat = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:

                        inv_covmat = np.linalg.pinv(cov_matrix)
                    
                    mv_df['mahalanobis_dist'] = np.zeros(len(mv_df))
                    for idx in range(len(X)):
                        diff = X[idx] - mean_vec
                        mv_df.loc[mv_df.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat).dot(diff))

                    threshold = chi2.ppf(0.975, df=len(available_columns))
                    mv_df['multivar_outlier'] = mv_df['mahalanobis_dist'] > threshold

                    multivar_outliers = mv_df[mv_df['multivar_outlier']].copy()
                    if len(multivar_outliers) > 0:

                        multivar_outliers['OutlierValue'] = multivar_outliers.apply(
                            lambda row: {col: row[col] for col in available_columns}, axis=1
                        ).apply(str)

                        multivar_outliers['analyzed_column'] = 'multiple'
                        
                        multivar_outliers['level'] = 'county'
                        multivar_outliers['outlier_type'] = 'multivariate'
                        multivar_outliers['entity_id'] = multivar_outliers[id_column]
                        multivar_outliers['variables_used'] = ', '.join(available_columns)
                        multivar_outliers['test_statistic_value'] = multivar_outliers['mahalanobis_dist']
                        multivar_outliers['test_type'] = 'Mahalanobis'

                        multivar_outliers['reference_mean'] = str(county_means)
                        multivar_outliers['reference_std'] = str(county_stds)
                        multivar_outliers['population_mean'] = str(county_means)
                        multivar_outliers['population_std'] = str(county_stds)

                        all_outliers.append(multivar_outliers)

                    if 'WardId' in mv_df.columns:
                        for ward in mv_df['WardId'].unique():
                            ward_data = mv_df[mv_df['WardId'] == ward].copy()

                            if len(ward_data) < 10:
                                continue

                            ward_means = {col: ward_data[col].mean() for col in available_columns}
                            ward_stds = {col: ward_data[col].std() for col in available_columns}

                            X_ward = ward_data[available_columns].values
                            if len(X_ward) > len(available_columns):  
                                try:
                                    mean_vec_ward = np.mean(X_ward, axis=0)
                                    cov_matrix_ward = np.cov(X_ward, rowvar=False)

                                    try:
                                        inv_covmat_ward = np.linalg.inv(cov_matrix_ward)
                                    except np.linalg.LinAlgError:
                                        inv_covmat_ward = np.linalg.pinv(cov_matrix_ward)
                                    
                                    ward_data['mahalanobis_dist'] = np.zeros(len(ward_data))
                                    for idx in range(len(X_ward)):
                                        diff = X_ward[idx] - mean_vec_ward
                                        ward_data.loc[ward_data.index[idx], 'mahalanobis_dist'] = np.sqrt(diff.dot(inv_covmat_ward).dot(diff))

                                    ward_data['multivar_outlier'] = ward_data['mahalanobis_dist'] > threshold

                                    ward_mv_outliers = ward_data[ward_data['multivar_outlier']].copy()
                                    if len(ward_mv_outliers) > 0:

                                        ward_mv_outliers['OutlierValue'] = ward_mv_outliers.apply(
                                            lambda row: {col: row[col] for col in available_columns}, axis=1
                                        ).apply(str)

                                        ward_mv_outliers['analyzed_column'] = 'multiple'
                                        
                                        ward_mv_outliers['level'] = 'ward'
                                        ward_mv_outliers['outlier_type'] = 'multivariate'
                                        ward_mv_outliers['entity_id'] = ward_mv_outliers[id_column]  
                                        ward_mv_outliers['variables_used'] = ', '.join(available_columns)
                                        ward_mv_outliers['test_statistic_value'] = ward_mv_outliers['mahalanobis_dist']
                                        ward_mv_outliers['test_type'] = 'Mahalanobis'

                                        ward_mv_outliers['reference_mean'] = str(ward_means)
                                        ward_mv_outliers['reference_std'] = str(ward_stds)
                                        ward_mv_outliers['population_mean'] = str(county_means)
                                        ward_mv_outliers['population_std'] = str(county_stds)

                                        all_outliers.append(ward_mv_outliers)
                                except Exception as e:

                                    if verbose:
                                        print(f"Error in ward {ward} multivariate analysis: {str(e)}")
            except Exception as e:

                print(f"Error in multivariate analysis: {str(e)}")
        
        # Combine all outliers into a single DataFrame
        if all_outliers:
            combined_outliers = pd.concat(all_outliers, ignore_index=True)
            

            columns_to_drop = ['multivar_outlier', 'z_score']
            combined_outliers = combined_outliers.drop([col for col in columns_to_drop if col in combined_outliers.columns], axis=1)
            
            for col in value_columns:
                if col in combined_outliers.columns:
                    combined_outliers = combined_outliers.drop(col, axis=1)

            if 'entity_id' in combined_outliers.columns:
                combined_outliers = combined_outliers.rename(columns={'entity_id': 'IndicatorType'})

            if id_column in combined_outliers.columns:
                combined_outliers = combined_outliers.drop(id_column, axis=1)

            if 'analyzed_column' not in combined_outliers.columns:
                print("Warning: analyzed_column field is missing from the results")

            results['outliers'] = combined_outliers
        
        return results


        
    def detect_outliers_Copying_Strategies(self, coping_df, id_column='CopyingStrategyId', 
                                   value_columns=['NumOfCopingDays'],
                                   z_threshold=2.5,
                                   verbose=False):
        
        # Check if dataframe is empty
        if coping_df.empty:
            return {
                'dataset_info': {
                    'total_records': 0,
                    'unique_wards': 0,
                    'unique_entities': 0,
                    'unique_households': 0
                },
                'column_stats': {},
                'outliers': pd.DataFrame(),
                'error': 'Input dataframe is empty'
            }

        results = {
            'dataset_info': {},
            'column_stats': {},
            'outliers': pd.DataFrame()
        }

        # 1. Dataset Overview
        results['dataset_info'] = {
            'total_records': len(coping_df),
            'unique_wards': coping_df['WardId'].nunique() if 'WardId' in coping_df.columns else 0,
            'unique_entities': coping_df[id_column].nunique(),
            'unique_households': coping_df['HouseHoldId'].nunique() if 'HouseHoldId' in coping_df.columns else 0
        }
        
        # 2. Check stats for each column
        for column in value_columns:
            # Skip if column doesn't exist in dataframe
            if column not in coping_df.columns:
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
        
        # Lists to store all types of outliers
        all_outliers = []
        
        # 3. Process each measurement column - univariate outliers
        for value_column in value_columns:
            if value_column not in coping_df.columns:
                continue
            
            zero_var_entities = []
            valid_entities = []
            
            # Calculate county-level mean and std dev for this column
            county_mean = coping_df[value_column].mean()
            county_std = coping_df[value_column].std()
            
            # 3.1 Check for zero variance entities in this column
            for entity_id in coping_df[id_column].unique():
                entity_data = coping_df[coping_df[id_column] == entity_id][value_column]
                # Skip if all values are null
                if entity_data.isnull().all():
                    continue
                    
                # Handling nulls by dropping them for variance check
                entity_data = entity_data.dropna()
                
                if len(entity_data) == 0 or entity_data.std() == 0:
                    zero_var_entities.append(entity_id)
                else:
                    valid_entities.append(entity_id)
            
            # 3.2 Ward-level and County-level Univariate Outlier Detection for this column
            for entity_id in valid_entities:

                entity_data = coping_df[coping_df[id_column] == entity_id].copy()

                entity_data = entity_data.dropna(subset=[value_column])
                
                if len(entity_data) <= 1:
                    continue

                entity_mean = entity_data[value_column].mean()
                entity_std = entity_data[value_column].std()
                
                entity_data['z_score'] = np.abs(stats.zscore(entity_data[value_column]))

                county_outliers = entity_data[entity_data['z_score'] > z_threshold].copy()
                if len(county_outliers) > 0:

                    county_outliers['analyzed_column'] = value_column

                    county_outliers['OutlierValue'] = county_outliers[value_column]
                    
                    county_outliers['level'] = 'county'
                    county_outliers['outlier_type'] = 'univariate'
                    county_outliers['entity_id'] = entity_id
                    county_outliers['test_statistic_value'] = county_outliers['z_score']
                    county_outliers['test_type'] = 'Z-score'

                    county_outliers['reference_mean'] = entity_mean
                    county_outliers['reference_std'] = entity_std
                    county_outliers['population_mean'] = county_mean
                    county_outliers['population_std'] = county_std

                    all_outliers.append(county_outliers)

                if 'WardId' in coping_df.columns:
                    for ward in coping_df['WardId'].unique():
                        ward_data = entity_data[entity_data['WardId'] == ward].copy()

                        if len(ward_data) <= 1 or ward_data[value_column].isnull().all():
                            continue

                        ward_mean = coping_df[coping_df['WardId'] == ward][value_column].mean()
                        ward_std = coping_df[coping_df['WardId'] == ward][value_column].std()

                        entity_ward_mean = ward_data[value_column].mean()
                        entity_ward_std = ward_data[value_column].std()

                        if len(ward_data) > 1 and ward_data[value_column].std() > 0:
                            ward_data['z_score'] = np.abs(stats.zscore(ward_data[value_column]))
                            ward_outliers = ward_data[ward_data['z_score'] > z_threshold].copy()
                            
                            if len(ward_outliers) > 0:

                                ward_outliers['analyzed_column'] = value_column

                                ward_outliers['OutlierValue'] = ward_outliers[value_column]
                                
                                ward_outliers['level'] = 'ward'
                                ward_outliers['outlier_type'] = 'univariate'
                                ward_outliers['entity_id'] = entity_id
                                ward_outliers['test_statistic_value'] = ward_outliers['z_score']
                                ward_outliers['test_type'] = 'Z-score'

                                ward_outliers['reference_mean'] = entity_ward_mean
                                ward_outliers['reference_std'] = entity_ward_std
                                ward_outliers['population_mean'] = ward_mean
                                ward_outliers['population_std'] = ward_std

                                all_outliers.append(ward_outliers)
        
        # Combine all outliers into a single DataFrame
        if all_outliers:
            combined_outliers = pd.concat(all_outliers, ignore_index=True)
            
            columns_to_drop = ['z_score']
            combined_outliers = combined_outliers.drop([col for col in columns_to_drop if col in combined_outliers.columns], axis=1)
            
            for col in value_columns:
                if col in combined_outliers.columns:
                    combined_outliers = combined_outliers.drop(col, axis=1)

            if 'entity_id' in combined_outliers.columns:
                combined_outliers = combined_outliers.rename(columns={'entity_id': 'IndicatorType'})

            if id_column in combined_outliers.columns:
                combined_outliers = combined_outliers.drop(id_column, axis=1)

            if 'analyzed_column' not in combined_outliers.columns:
                print("Warning: analyzed_column field is missing from the results")

            results['outliers'] = combined_outliers
        
        return results



    def detect_outliers_crop_production(crop_df, id_column='CropId', 
                               value_columns=['AcresPlantedInLastFourWks', 'AcresHarvestedInLastFourWks', 
                                             'KgsHarvestedInLastFourWks', 'OwnProductionStockInKg', 
                                             'KgsSoldInLastFourWks', 'PricePerKg'],
                               z_threshold=2.5):
        
        # Check if dataframe is empty
        if coping_df.empty:
            return {
                'dataset_info': {
                    'total_records': 0,
                    'unique_wards': 0,
                    'unique_entities': 0,
                    'unique_households': 0
                },
                'column_stats': {},
                'outliers': pd.DataFrame(),
                'error': 'Input dataframe is empty'
            }

        results = {
            'dataset_info': {},
            'column_stats': {},
            'outliers': pd.DataFrame()
        }

        # 1. Dataset Overview
        results['dataset_info'] = {
            'total_records': len(coping_df),
            'unique_wards': coping_df['WardId'].nunique() if 'WardId' in coping_df.columns else 0,
            'unique_entities': coping_df[id_column].nunique(),
            'unique_households': coping_df['HouseHoldId'].nunique() if 'HouseHoldId' in coping_df.columns else 0
        }
        
        # 2. Check stats for each column
        for column in value_columns:
            # Skip if column doesn't exist in dataframe
            if column not in coping_df.columns:
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
        
        # Lists to store all types of outliers
        all_outliers = []
        
        # 3. Process each measurement column - univariate outliers
        for value_column in value_columns:
            if value_column not in coping_df.columns:
                continue
            
            zero_var_entities = []
            valid_entities = []
            
            # Calculate county-level mean and std dev for this column
            county_mean = coping_df[value_column].mean()
            county_std = coping_df[value_column].std()
            
            # 3.1 Check for zero variance entities in this column
            for entity_id in coping_df[id_column].unique():
                entity_data = coping_df[coping_df[id_column] == entity_id][value_column]
                # Skip if all values are null
                if entity_data.isnull().all():
                    continue
                    
                # Handling nulls by dropping them for variance check
                entity_data = entity_data.dropna()
                
                if len(entity_data) == 0 or entity_data.std() == 0:
                    zero_var_entities.append(entity_id)
                else:
                    valid_entities.append(entity_id)
            
            # 3.2 Ward-level and County-level Univariate Outlier Detection for this column
            for entity_id in valid_entities:

                entity_data = coping_df[coping_df[id_column] == entity_id].copy()

                entity_data = entity_data.dropna(subset=[value_column])
                
                if len(entity_data) <= 1:
                    continue

                entity_mean = entity_data[value_column].mean()
                entity_std = entity_data[value_column].std()
                
                entity_data['z_score'] = np.abs(stats.zscore(entity_data[value_column]))

                county_outliers = entity_data[entity_data['z_score'] > z_threshold].copy()
                if len(county_outliers) > 0:

                    county_outliers['analyzed_column'] = value_column

                    county_outliers['OutlierValue'] = county_outliers[value_column]
                    
                    county_outliers['level'] = 'county'
                    county_outliers['outlier_type'] = 'univariate'
                    county_outliers['entity_id'] = entity_id
                    county_outliers['test_statistic_value'] = county_outliers['z_score']
                    county_outliers['test_type'] = 'Z-score'

                    county_outliers['reference_mean'] = entity_mean
                    county_outliers['reference_std'] = entity_std
                    county_outliers['population_mean'] = county_mean
                    county_outliers['population_std'] = county_std

                    all_outliers.append(county_outliers)

                if 'WardId' in coping_df.columns:
                    for ward in coping_df['WardId'].unique():
                        ward_data = entity_data[entity_data['WardId'] == ward].copy()

                        if len(ward_data) <= 1 or ward_data[value_column].isnull().all():
                            continue

                        ward_mean = coping_df[coping_df['WardId'] == ward][value_column].mean()
                        ward_std = coping_df[coping_df['WardId'] == ward][value_column].std()

                        entity_ward_mean = ward_data[value_column].mean()
                        entity_ward_std = ward_data[value_column].std()

                        if len(ward_data) > 1 and ward_data[value_column].std() > 0:
                            ward_data['z_score'] = np.abs(stats.zscore(ward_data[value_column]))
                            ward_outliers = ward_data[ward_data['z_score'] > z_threshold].copy()
                            
                            if len(ward_outliers) > 0:

                                ward_outliers['analyzed_column'] = value_column

                                ward_outliers['OutlierValue'] = ward_outliers[value_column]
                                
                                ward_outliers['level'] = 'ward'
                                ward_outliers['outlier_type'] = 'univariate'
                                ward_outliers['entity_id'] = entity_id
                                ward_outliers['test_statistic_value'] = ward_outliers['z_score']
                                ward_outliers['test_type'] = 'Z-score'

                                ward_outliers['reference_mean'] = entity_ward_mean
                                ward_outliers['reference_std'] = entity_ward_std
                                ward_outliers['population_mean'] = ward_mean
                                ward_outliers['population_std'] = ward_std

                                all_outliers.append(ward_outliers)
        
        # Combine all outliers into a single DataFrame
        if all_outliers:
            combined_outliers = pd.concat(all_outliers, ignore_index=True)
            
            columns_to_drop = ['z_score']
            combined_outliers = combined_outliers.drop([col for col in columns_to_drop if col in combined_outliers.columns], axis=1)
            
            for col in value_columns:
                if col in combined_outliers.columns:
                    combined_outliers = combined_outliers.drop(col, axis=1)

            if 'entity_id' in combined_outliers.columns:
                combined_outliers = combined_outliers.rename(columns={'entity_id': 'IndicatorType'})

            if id_column in combined_outliers.columns:
                combined_outliers = combined_outliers.drop(id_column, axis=1)

            if 'analyzed_column' not in combined_outliers.columns:
                print("Warning: analyzed_column field is missing from the results")

            results['outliers'] = combined_outliers
        
        return results





    def detect_outliers_Food_Consumption(food_c_df, id_column='FoodTypeId', 
                               value_columns=['NumDaysEaten'],
                               z_threshold=2.5):

        
        # Check if dataframe is empty
        if coping_df.empty:
            return {
                'dataset_info': {
                    'total_records': 0,
                    'unique_wards': 0,
                    'unique_entities': 0,
                    'unique_households': 0
                },
                'column_stats': {},
                'outliers': pd.DataFrame(),
                'error': 'Input dataframe is empty'
            }

        results = {
            'dataset_info': {},
            'column_stats': {},
            'outliers': pd.DataFrame()
        }

        # 1. Dataset Overview
        results['dataset_info'] = {
            'total_records': len(coping_df),
            'unique_wards': coping_df['WardId'].nunique() if 'WardId' in coping_df.columns else 0,
            'unique_entities': coping_df[id_column].nunique(),
            'unique_households': coping_df['HouseHoldId'].nunique() if 'HouseHoldId' in coping_df.columns else 0
        }
        
        # 2. Check stats for each column
        for column in value_columns:
            # Skip if column doesn't exist in dataframe
            if column not in coping_df.columns:
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
        
        # Lists to store all types of outliers
        all_outliers = []
        
        # 3. Process each measurement column - univariate outliers
        for value_column in value_columns:
            if value_column not in coping_df.columns:
                continue
            
            zero_var_entities = []
            valid_entities = []
            
            # Calculate county-level mean and std dev for this column
            county_mean = coping_df[value_column].mean()
            county_std = coping_df[value_column].std()
            
            # 3.1 Check for zero variance entities in this column
            for entity_id in coping_df[id_column].unique():
                entity_data = coping_df[coping_df[id_column] == entity_id][value_column]
                # Skip if all values are null
                if entity_data.isnull().all():
                    continue
                    
                # Handling nulls by dropping them for variance check
                entity_data = entity_data.dropna()
                
                if len(entity_data) == 0 or entity_data.std() == 0:
                    zero_var_entities.append(entity_id)
                else:
                    valid_entities.append(entity_id)
            
            # 3.2 Ward-level and County-level Univariate Outlier Detection for this column
            for entity_id in valid_entities:

                entity_data = coping_df[coping_df[id_column] == entity_id].copy()

                entity_data = entity_data.dropna(subset=[value_column])
                
                if len(entity_data) <= 1:
                    continue

                entity_mean = entity_data[value_column].mean()
                entity_std = entity_data[value_column].std()
                
                entity_data['z_score'] = np.abs(stats.zscore(entity_data[value_column]))

                county_outliers = entity_data[entity_data['z_score'] > z_threshold].copy()
                if len(county_outliers) > 0:

                    county_outliers['analyzed_column'] = value_column

                    county_outliers['OutlierValue'] = county_outliers[value_column]
                    
                    county_outliers['level'] = 'county'
                    county_outliers['outlier_type'] = 'univariate'
                    county_outliers['entity_id'] = entity_id
                    county_outliers['test_statistic_value'] = county_outliers['z_score']
                    county_outliers['test_type'] = 'Z-score'

                    county_outliers['reference_mean'] = entity_mean
                    county_outliers['reference_std'] = entity_std
                    county_outliers['population_mean'] = county_mean
                    county_outliers['population_std'] = county_std

                    all_outliers.append(county_outliers)

                if 'WardId' in coping_df.columns:
                    for ward in coping_df['WardId'].unique():
                        ward_data = entity_data[entity_data['WardId'] == ward].copy()

                        if len(ward_data) <= 1 or ward_data[value_column].isnull().all():
                            continue

                        ward_mean = coping_df[coping_df['WardId'] == ward][value_column].mean()
                        ward_std = coping_df[coping_df['WardId'] == ward][value_column].std()

                        entity_ward_mean = ward_data[value_column].mean()
                        entity_ward_std = ward_data[value_column].std()

                        if len(ward_data) > 1 and ward_data[value_column].std() > 0:
                            ward_data['z_score'] = np.abs(stats.zscore(ward_data[value_column]))
                            ward_outliers = ward_data[ward_data['z_score'] > z_threshold].copy()
                            
                            if len(ward_outliers) > 0:

                                ward_outliers['analyzed_column'] = value_column

                                ward_outliers['OutlierValue'] = ward_outliers[value_column]
                                
                                ward_outliers['level'] = 'ward'
                                ward_outliers['outlier_type'] = 'univariate'
                                ward_outliers['entity_id'] = entity_id
                                ward_outliers['test_statistic_value'] = ward_outliers['z_score']
                                ward_outliers['test_type'] = 'Z-score'

                                ward_outliers['reference_mean'] = entity_ward_mean
                                ward_outliers['reference_std'] = entity_ward_std
                                ward_outliers['population_mean'] = ward_mean
                                ward_outliers['population_std'] = ward_std

                                all_outliers.append(ward_outliers)
        
        # Combine all outliers into a single DataFrame
        if all_outliers:
            combined_outliers = pd.concat(all_outliers, ignore_index=True)
            
            columns_to_drop = ['z_score']
            combined_outliers = combined_outliers.drop([col for col in columns_to_drop if col in combined_outliers.columns], axis=1)
            
            for col in value_columns:
                if col in combined_outliers.columns:
                    combined_outliers = combined_outliers.drop(col, axis=1)

            if 'entity_id' in combined_outliers.columns:
                combined_outliers = combined_outliers.rename(columns={'entity_id': 'IndicatorType'})

            if id_column in combined_outliers.columns:
                combined_outliers = combined_outliers.drop(id_column, axis=1)

            if 'analyzed_column' not in combined_outliers.columns:
                print("Warning: analyzed_column field is missing from the results")

            results['outliers'] = combined_outliers
        
        return results




    def detect_outliers_livestock_production(livestock_df, id_column='AnimalId', 
                               value_columns=['NumberKeptToday', 'NumberBornInLastFourWeeks', 
                                             'NumberPurchasedInLastFourWeeks', 'NumberSoldInLastFourWeeks', 
                                             'AveragePricePerAnimalSold', 'NumberDiedDuringLastFourWeeks'],
                               z_threshold=2.5):
        
        # Check if dataframe is empty
        if coping_df.empty:
            return {
                'dataset_info': {
                    'total_records': 0,
                    'unique_wards': 0,
                    'unique_entities': 0,
                    'unique_households': 0
                },
                'column_stats': {},
                'outliers': pd.DataFrame(),
                'error': 'Input dataframe is empty'
            }

        results = {
            'dataset_info': {},
            'column_stats': {},
            'outliers': pd.DataFrame()
        }

        # 1. Dataset Overview
        results['dataset_info'] = {
            'total_records': len(coping_df),
            'unique_wards': coping_df['WardId'].nunique() if 'WardId' in coping_df.columns else 0,
            'unique_entities': coping_df[id_column].nunique(),
            'unique_households': coping_df['HouseHoldId'].nunique() if 'HouseHoldId' in coping_df.columns else 0
        }
        
        # 2. Check stats for each column
        for column in value_columns:
            # Skip if column doesn't exist in dataframe
            if column not in coping_df.columns:
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
        
        # Lists to store all types of outliers
        all_outliers = []
        
        # 3. Process each measurement column - univariate outliers
        for value_column in value_columns:
            if value_column not in coping_df.columns:
                continue
            
            zero_var_entities = []
            valid_entities = []
            
            # Calculate county-level mean and std dev for this column
            county_mean = coping_df[value_column].mean()
            county_std = coping_df[value_column].std()
            
            # 3.1 Check for zero variance entities in this column
            for entity_id in coping_df[id_column].unique():
                entity_data = coping_df[coping_df[id_column] == entity_id][value_column]
                # Skip if all values are null
                if entity_data.isnull().all():
                    continue
                    
                # Handling nulls by dropping them for variance check
                entity_data = entity_data.dropna()
                
                if len(entity_data) == 0 or entity_data.std() == 0:
                    zero_var_entities.append(entity_id)
                else:
                    valid_entities.append(entity_id)
            
            # 3.2 Ward-level and County-level Univariate Outlier Detection for this column
            for entity_id in valid_entities:

                entity_data = coping_df[coping_df[id_column] == entity_id].copy()

                entity_data = entity_data.dropna(subset=[value_column])
                
                if len(entity_data) <= 1:
                    continue

                entity_mean = entity_data[value_column].mean()
                entity_std = entity_data[value_column].std()
                
                entity_data['z_score'] = np.abs(stats.zscore(entity_data[value_column]))

                county_outliers = entity_data[entity_data['z_score'] > z_threshold].copy()
                if len(county_outliers) > 0:

                    county_outliers['analyzed_column'] = value_column

                    county_outliers['OutlierValue'] = county_outliers[value_column]
                    
                    county_outliers['level'] = 'county'
                    county_outliers['outlier_type'] = 'univariate'
                    county_outliers['entity_id'] = entity_id
                    county_outliers['test_statistic_value'] = county_outliers['z_score']
                    county_outliers['test_type'] = 'Z-score'

                    county_outliers['reference_mean'] = entity_mean
                    county_outliers['reference_std'] = entity_std
                    county_outliers['population_mean'] = county_mean
                    county_outliers['population_std'] = county_std

                    all_outliers.append(county_outliers)

                if 'WardId' in coping_df.columns:
                    for ward in coping_df['WardId'].unique():
                        ward_data = entity_data[entity_data['WardId'] == ward].copy()

                        if len(ward_data) <= 1 or ward_data[value_column].isnull().all():
                            continue

                        ward_mean = coping_df[coping_df['WardId'] == ward][value_column].mean()
                        ward_std = coping_df[coping_df['WardId'] == ward][value_column].std()

                        entity_ward_mean = ward_data[value_column].mean()
                        entity_ward_std = ward_data[value_column].std()

                        if len(ward_data) > 1 and ward_data[value_column].std() > 0:
                            ward_data['z_score'] = np.abs(stats.zscore(ward_data[value_column]))
                            ward_outliers = ward_data[ward_data['z_score'] > z_threshold].copy()
                            
                            if len(ward_outliers) > 0:

                                ward_outliers['analyzed_column'] = value_column

                                ward_outliers['OutlierValue'] = ward_outliers[value_column]
                                
                                ward_outliers['level'] = 'ward'
                                ward_outliers['outlier_type'] = 'univariate'
                                ward_outliers['entity_id'] = entity_id
                                ward_outliers['test_statistic_value'] = ward_outliers['z_score']
                                ward_outliers['test_type'] = 'Z-score'

                                ward_outliers['reference_mean'] = entity_ward_mean
                                ward_outliers['reference_std'] = entity_ward_std
                                ward_outliers['population_mean'] = ward_mean
                                ward_outliers['population_std'] = ward_std

                                all_outliers.append(ward_outliers)
        
        # Combine all outliers into a single DataFrame
        if all_outliers:
            combined_outliers = pd.concat(all_outliers, ignore_index=True)
            
            columns_to_drop = ['z_score']
            combined_outliers = combined_outliers.drop([col for col in columns_to_drop if col in combined_outliers.columns], axis=1)
            
            for col in value_columns:
                if col in combined_outliers.columns:
                    combined_outliers = combined_outliers.drop(col, axis=1)

            if 'entity_id' in combined_outliers.columns:
                combined_outliers = combined_outliers.rename(columns={'entity_id': 'IndicatorType'})

            if id_column in combined_outliers.columns:
                combined_outliers = combined_outliers.drop(id_column, axis=1)

            if 'analyzed_column' not in combined_outliers.columns:
                print("Warning: analyzed_column field is missing from the results")

            results['outliers'] = combined_outliers
        
        return results

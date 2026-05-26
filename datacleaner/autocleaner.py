import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class AutoCleaner:
    def __init__(self, target_column=None, outlier_threshold=1.5):
        self.target_column = target_column
        self.outlier_threshold = outlier_threshold
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        #work on a copy ( don't overwrite the original data )
        data = df.copy()
        
        #separate target column if specified so it remains untouched
        target_series = None
        if self.target_column and self.target_column in data.columns:
            target_series = data[self.target_column]
            data = data.drop(columns=[self.target_column])
            
        #identify column types
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        #handling missing values
        for col in num_cols:
            if data[col].isnull().sum() > 0:
                #filling the missing values
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
                
        for col in cat_cols:
            if data[col].isnull().sum() > 0:
                mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_value)
                
        #iqr
        for col in num_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            
            #cap values outside the boundaries
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

       #filter columns
        #Drops columns like random 'IDs' or 'Names' which break ML models
        valid_cat_cols = []
        for col in cat_cols:
            unique_count = data[col].nunique()
            #If a text column has too many unique values relative to row count, drop it
            if unique_count > (len(data) * 0.5) and unique_count > 10:
                print(f"Dropping high-cardinality column: {col}")
            else:
                valid_cat_cols.append(col)

        if valid_cat_cols:
            #use One-Hot Encoding and convert directly back to a readable DataFrame
            encoded_data = pd.get_dummies(data[valid_cat_cols], drop_first=True)
            data = data.drop(columns=cat_cols) # Drop original text columns
            data = pd.concat([data, encoded_data], axis=1) # Join encoded metrics

        #feature scaling
        #update our numerical columns list as categories are gone now
        final_num_cols = [c for c in num_cols if c in data.columns]
        if final_num_cols:
            data[final_num_cols] = self.scaler.fit_transform(data[final_num_cols])
            
        #re-attach target column if we extracted it earlier
        if target_series is not None:
            data[self.target_column] = target_series.values
            
        return data
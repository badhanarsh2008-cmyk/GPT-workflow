import pandas as pd
import numpy as np
from autocleaner import AutoCleaner

#created a deliberately messy dataset ( for testing )
raw_data = pd.DataFrame({
    'PassengerId': ['ID_001', 'ID_002', 'ID_003', 'ID_004', 'ID_005'], #high cardinality
    'Age': [22, np.nan, 38, 150, 26],         #contains a missing value and a massive outlier (150)
    'Salary': [50000, 60000, np.nan, 80000, 52000], #missing value
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Male'], #missing categorical value
    'Purchased': [0, 1, 0, 1, 0]              #target label
})

print("RAW MESSY DATA ")
print(raw_data)
print("\n" + "="*40 + "\n")

#own module
cleaner = AutoCleaner(target_column='Purchased')

#run the single line optimization script
cleaned_data = cleaner.clean_data(raw_data)

print("=== CLEANED & PROCESSED DATA FOR ML ===")
print(cleaned_data)
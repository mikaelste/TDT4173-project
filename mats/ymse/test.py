
import pandas as pd

# Create two sample DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3']}
                   )

df2 = pd.DataFrame({'B': ['B0', 'B1', 'B2', 'B3']})
print(df2)

# Merge the DataFrames on their indices
merged_df = pd.merge(df1, df2, left_index=True, right_index=True)

print(merged_df)

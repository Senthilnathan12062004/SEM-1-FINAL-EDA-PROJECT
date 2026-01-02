import numpy as np

def clean_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    print("Data Cleaning Completed")
    return df

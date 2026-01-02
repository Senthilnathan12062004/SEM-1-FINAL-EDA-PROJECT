from sklearn.preprocessing import StandardScaler

def transform_data(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Data Transformation Completed")
    return df

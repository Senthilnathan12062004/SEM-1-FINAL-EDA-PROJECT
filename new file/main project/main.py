from data_import import load_data
from data_cleaning import clean_data
from transformation import transform_data
from stats_analysis import descriptive_stats
from visualization import visualize_data
from modeling import run_models

# Load data
df = load_data("car_insurance_premium_dataset.csv")

# Clean data
df = clean_data(df)

# Transform data
df = transform_data(df)

# Statistics
descriptive_stats(df)

# Visualization
visualize_data(df)

# Modeling
df = run_models(df)

print("\nEDA Project Completed Successfully")

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    numeric_cols = df.select_dtypes(include='number').columns

    df[numeric_cols[0]].hist()
    plt.title("Histogram")
    plt.show()

    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

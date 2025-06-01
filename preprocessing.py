import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


def preprocess(data):

    def plot_numerical_distributions(df, numerical_cols, show=False):
        """
        Shows histogram and skewness values of given numeric columns as subplots.

        Args:
            df (pd.DataFrame): Data framework.
            numerical_cols (list): List of numeric columns to be plotted histograms.
            skew_columns (list): list of columns with skewness not between -0.5 and 0.5
            show (bool): show the skewness grafice
        """
        skew_columns = []
        num_cols = len(numerical_cols)
        num_rows = (num_cols + 1) // 2  # Set the number of rows
        fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))
        axes = axes.flatten()  # Convert 2-dimensional axis array to one dimensional

        for i, col in enumerate(numerical_cols):
            sns.histplot(x=df[col].dropna(), kde=True, bins=50, ax=axes[i])
            axes[i].set_title(f"{col} Distribution")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

            sk = skew(df[col].dropna())
            print(f"Skewness for {col}: {sk:.2f}")

            if sk > 1 or sk < -1:
                skew_columns.append(col)

        # Clear unused subplots
        if num_cols % 2 != 0:
            fig.delaxes(axes[-1])

        if show:
            plt.tight_layout()
            plt.show()
        return skew_columns
    def apply_log_transformation(df, columns):
        for col in columns:
            if (df[col] <= -1).any():
                print(f"{col} is not convenient for log1 conversion  (there are negative variables).")
                continue
            df['NEW_' + col + '_log'] = np.log1p(df[col])
            print(f"{col} conversion is done .")


    numerical_col = [col for col in data.columns if data[col].dtype != 'O']
    skew_list = plot_numerical_distributions(data, numerical_col)

    ########################
    # Feature Engineering & Feature Extraction
    ########################
    data['New_HeartRateMinute'] = data['Heart_Rate'] / data['Duration']

    data['New_HeartRateDuration'] = data['Heart_Rate'] * data['Duration']

    data['New_DurationBodyTemp'] = data['Duration'] * data['Body_Temp']

    data['New_AgeHeartRate'] = data['Age'] * data['Heart_Rate']

    data['New_AgeBodyTemp'] = data['Age'] * data['Body_Temp']

    data['New_BodyArea'] = data['Height'] * data['Weight']

    data['New_DurationCategory'] = pd.qcut(data['Duration'], q=4,
                                               labels=['Sedentary', 'Lightly_Active', 'Moderately_Active',
                                                       'Very_Active'])

    bins = [20, 30, 45, 60, 70, 80]
    labels = ['Young', 'Adult', 'Middle_Aged', 'Old_Age', 'Old']
    data['New_AgeCategory'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    ########################
    # Encoding & Scaling
    ########################

    def label_encoder(dataframe, binary_col):
        labelencoder = LabelEncoder()
        dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
        return dataframe

    def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    label_encoder(data, 'Sex')

    categorical_cols = [col for col in data.columns if 2 < data[col].nunique() < 10]
    data = one_hot_encoder(data, categorical_cols, drop_first=True)

    scaler = StandardScaler()
    scale_cols = [col for col in data.columns if data[col].nunique() > 20 and data[col].dtypes != 'O']


    data[scale_cols] = scaler.fit_transform(data[scale_cols])

    return data

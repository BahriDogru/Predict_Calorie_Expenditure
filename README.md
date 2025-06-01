# Calories Prediction Project (Kaggle Playground Series - S5E5)

## Project Overview

This project was developed for participation in the [Kaggle Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5) machine learning competition. The primary goal of the competition is to accurately predict the amount of calories burned based on various physical activity and personal health metrics.

### Dataset

The dataset used in this project is provided by the Kaggle competition and includes the following features:
- **`id`**: Unique identifier
- **`Age`**: Age (years)
- **`Height`**: Height (cm)
- **`Weight`**: Weight (kg)
- **`Duration`**: Activity duration (minutes)
- **`Heart_Rate`**: Heart rate (bpm)
- **`Body_Temp`**: Body temperature (Celsius)
- **`Calories`**: Calories burned (target variable)

## Project Structure

The project primarily consists of two Python files:
- `main.py`: Contains data loading, exploratory data analysis (EDA), feature engineering, modeling, hyperparameter optimization, and prediction generation steps.
- `preprocessing.py`: A separate module called within `main.py`, containing data preprocessing steps (feature engineering, encoding, scaling). This structure ensures that consistent transformations are applied to both the training and test datasets.

## Workflow

The project involves the following steps:

1.  **Data Loading**: `train.csv` and `test.csv` files are loaded into pandas DataFrames.
2.  **Exploratory Data Analysis (EDA)**:
    * General information about the datasets (`info()`, `isna().sum()`) is inspected.
    * Statistical summaries of numerical features (`describe()`) are examined.
    * Distributions of numerical features (histograms and skewness values) are visualized.
    * A correlation matrix between numerical features is plotted.
3.  **Feature Engineering**:
    * New meaningful features are derived from existing ones (e.g., `New_HeartRateMinute`, `New_BodyArea`, `New_DurationCategory`, `New_AgeCategory`). These new features aim to enhance the model's predictive power.
4.  **Encoding & Scaling**:
    * Categorical features (`Sex`, `New_DurationCategory`, `New_AgeCategory`) are converted to numerical format using Label Encoding and One-Hot Encoding.
    * Numerical features are scaled using `StandardScaler`.
5.  **Modeling**:
    * Various regression models (Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Bagging Regressor, XGBoost, LightGBM, CatBoost) are compared with initial parameters.
    * Log RMSE (logarithmic Root Mean Squared Error) is used as the model evaluation metric.
    * The `compare_models` function is used to compare cross-validation scores of the models. CatBoost shows the best performance in this comparison.
6.  **Hyperparameter Optimization**:
    * Hyperparameter optimization is performed for the best-performing CatBoost model using the Optuna library. This aims to further improve the model's performance.
    * The final model is trained using the best hyperparameters obtained from the optimization.
7.  **Validation and Prediction**:
    * The final model's performance on the validation set is evaluated using Log RMSE.
    * Predictions are made on the preprocessed test dataset.
    * The prediction results are saved to a `submission.csv` file, formatted for Kaggle submission. Negative predictions are rounded up to zero.

## Setup and Running

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```
2.  **Install Required Libraries**:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn optuna lightgbm catboost xgboost
    ```
3.  **Download the Dataset**:
    Download the `train.csv` and `test.csv` files from the Kaggle competition page ([https://www.kaggle.com/competitions/playground-series-s5e5](https://www.kaggle.com/competitions/playground-series-s5e5)). Place these files inside a folder named `dataset/` in your project's root directory.
    ```
    .
    ├── main.py
    ├── preprocessing.py
    ├── dataset/
    │   ├── train.csv
    │   └── test.csv
    └── README.md
    ```
4.  **Run the Code**:
    ```bash
    python main.py
    ```
    This command will train the model, make predictions, and generate the `submission.csv` file.

## Results

The CatBoost model, after hyperparameter optimization with Optuna, showed the best performance. The Log RMSE score on the validation set was recorded as: `0.0606966032352811`.

## Future Improvements

* Further feature engineering techniques can be explored.
* Different ensemble methods can be investigated.
* Optuna optimization can be enhanced with more comprehensive hyperparameter search spaces and a higher number of trials.
* Outlier detection and handling can be implemented.

## License

This project is licensed under the MIT License.
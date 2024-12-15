from typing import List, Dict, Any 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import polars as pl
import os

class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.

        Args:
            train_data_path (str): Path to the training dataset CSV file.
            test_data_path (str): Path to the testing dataset CSV file.
        """
        self.train_data = pl.read_csv(train_data_path, null_values=["NA"])
        self.test_data = pl.read_csv(test_data_path, null_values=["NA"])

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning using Polars.
        """
        print("Starting data cleaning...")

        # Remove leading/trailing spaces from column names
        self.train_data.columns = [col.strip() for col in self.train_data.columns]
        self.test_data.columns = [col.strip() for col in self.test_data.columns]

        # Separate numeric and categorical columns in the training data
        numeric_columns = [col for col in self.train_data.columns if self.train_data[col].dtype in [pl.Float64, pl.Int64]]
        categorical_columns = [col for col in self.train_data.columns if self.train_data[col].dtype == pl.Utf8]

        # Fill missing values in numeric columns with the mean for the training data
        self.train_data = self.train_data.with_columns([
            pl.when(pl.col(col).is_null()).then(self.train_data[col].mean()).otherwise(pl.col(col)).alias(col)
            for col in numeric_columns
        ])

        # Fill missing values in categorical columns with the mode for the training data
        self.train_data = self.train_data.with_columns([
            pl.when(pl.col(col).is_null()).then(pl.lit(self.train_data[col].mode().first())).otherwise(pl.col(col)).alias(col)
            for col in categorical_columns
        ])

        # Create dummy variables (one-hot encoding) for categorical columns in the training data
        for col in categorical_columns:
            unique_values = self.train_data[col].unique().to_list()  # Get the unique categories for the column
            for value in unique_values:
                # Create a new column indicating whether the value exists
                self.train_data = self.train_data.with_columns([
                    (pl.col(col) == value).cast(pl.Int8).alias(f"{col}_{value}")
                ])

        # Handle missing values for the test dataset (without SalePrice column)
        # Only fill missing values for numeric and categorical columns excluding 'SalePrice'
        test_numeric_columns = [col for col in self.test_data.columns if self.test_data[col].dtype in [pl.Float64, pl.Int64]]
        test_categorical_columns = [col for col in self.test_data.columns if self.test_data[col].dtype == pl.Utf8]

        # Fill missing values in numeric columns with the mean for the test data
        self.test_data = self.test_data.with_columns([
            pl.when(pl.col(col).is_null()).then(self.test_data[col].mean()).otherwise(pl.col(col)).alias(col)
            for col in test_numeric_columns
        ])

        # Fill missing values in categorical columns with the mode for the test data
        self.test_data = self.test_data.with_columns([
            pl.when(pl.col(col).is_null()).then(pl.lit(self.test_data[col].mode().first())).otherwise(pl.col(col)).alias(col)
            for col in test_categorical_columns
        ])

        # Create dummy variables (one-hot encoding) for categorical columns in the test data
        for col in test_categorical_columns:
            unique_values = self.test_data[col].unique().to_list()  # Get the unique categories for the column
            for value in unique_values:
                # Create a new column indicating whether the value exists
                self.test_data = self.test_data.with_columns([
                    (pl.col(col) == value).cast(pl.Int8).alias(f"{col}_{value}")
                ])

        # Assign cleaned data to the attribute
        self.train_data_clean = self.train_data
        self.test_data_clean = self.test_data

        print(f"Data cleaned successfully. Remaining rows in training data: {self.train_data_clean.height}")
        print(f"Data cleaned successfully. Remaining rows in testing data: {self.test_data_clean.height}")

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable,
        and preprocessing them for training and testing.

        Args:
            target_column (str): Name of the target variable column. Default is 'SalePrice'.
            selected_predictors (List[str]): Specific columns to use as predictors.
                If None, use all columns except the target.

        Returns:
            Tuple: X_train, X_test, y_train
        """
        # Ensure only shared columns between train and test are selected
        shared_columns = list(set(self.train_data.columns) & set(self.test_data.columns))
        
        if selected_predictors is None:
            # Use all shared columns except the target in train, but drop SalePrice if in test
            selected_predictors = [col for col in shared_columns if col != target_column]

        X_train = self.train_data.select(selected_predictors)
        y_train = self.train_data.select(target_column)
        X_test = self.test_data.select(selected_predictors)

        numeric_columns = [col for col in selected_predictors if X_train[col].dtype in [pl.Float64, pl.Int64]]
        categorical_columns = [col for col in selected_predictors if X_train[col].dtype == pl.Utf8]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )

        X_train_transformed = self.preprocessor.fit_transform(X_train.to_pandas())
        X_test_transformed = self.preprocessor.transform(X_test.to_pandas())
        y_train_flat = y_train.to_pandas().values.flatten()

        return X_train_transformed, X_test_transformed, y_train_flat

    def train_baseline_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate baseline machine learning models.

        Returns:
            Dict[str, Dict[str, Any]]: Model evaluation metrics.
        """
        X_train, _, y_train = self.prepare_features()  # We only need the train split here
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }

        results = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            model.fit(X_train_split, y_train_split)

            y_train_pred = model.predict(X_train_split)
            y_val_pred = model.predict(X_val)

            metrics = {
                "Train MSE": mean_squared_error(y_train_split, y_train_pred),
                "Train MAE": mean_absolute_error(y_train_split, y_train_pred),
                "Train R2": r2_score(y_train_split, y_train_pred),
                "Train MAPE": mean_absolute_percentage_error(y_train_split, y_train_pred),
                "Val MSE": mean_squared_error(y_val, y_val_pred),
                "Val MAE": mean_absolute_error(y_val, y_val_pred),
                "Val R2": r2_score(y_val, y_val_pred),
                "Val MAPE": mean_absolute_percentage_error(y_val, y_val_pred),
            }

            results[model_name] = {"metrics": metrics}

        return results

    def forecast_sales_price(self, model_type: str = 'Linear Regression'):
        """
        Generate house price predictions for the test dataset and save to a CSV file.

        Args:
            model_type (str): Type of model to use ('Linear Regression' or 'Random Forest Regressor').
        """
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Random Forest Regressor':
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        X_train, X_test, y_train = self.prepare_features()

        print(f"Training {model_type} for forecasting...")
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)

        submission_df = self.test_data.select(['Id']).to_pandas()
        submission_df['SalePrice'] = y_test_pred

        output_dir = "real_estate_toolkit/src/real_estate_toolkit/ml_models/outputs"
        os.makedirs(output_dir, exist_ok=True)
        submission_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)

        print(f"Predictions saved successfully to {output_dir}/submission.csv")

    

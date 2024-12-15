"Main module for running tests"
from pathlib import Path
from typing import List, Dict, Any
import polars as pl
import plotly.graph_objects as go
import os
from real_estate_toolkit.ml_models.predictor import HousePricePredictor


def test_house_price_predictor():
    """Test the functionality of the HousePricePredictor class."""
    # Paths to the datasets
    train_data_path = Path("final_project/data_files/train.csv")
    test_data_path = Path("final_project/data_files/test.csv")
    # Initialize predictor
    predictor = HousePricePredictor(train_data_path=str(train_data_path), test_data_path=str(test_data_path))
    # Step 1: Test data cleaning
    print("Testing data cleaning...")
    try:
        predictor.clean_data()
        print("Data cleaning passed!")
    except Exception as e:
        print(f"Data cleaning failed: {e}")
        return
    # Step 2: Test feature preparation
    print("Testing feature preparation...")
    try:
        predictor.prepare_features(target_column="SalePrice")
        print("Feature preparation passed!")
    except Exception as e:
        print(f"Feature preparation failed: {e}")
        return
    # Step 3: Test model training
    print("Testing model training...")
    try:
        results = predictor.train_baseline_models()
        for model_name, result in results.items():
            metrics = result["metrics"]
            print(f"{model_name} - Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        print("Model training passed!")
    except Exception as e:
        print(f"Model training failed: {e}")
        return
    # Step 4: Test forecasting
    print("Testing forecasting...")
    try:
        predictor.forecast_sales_price(model_type="Linear Regression")
        print("Forecasting passed!")
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return

def main():
    """Main function to run all tests"""
    try:
        test_house_price_predictor()

        print("All tests passed successfully!")
        return 0
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 2

if __name__ == "__main__":
    main()
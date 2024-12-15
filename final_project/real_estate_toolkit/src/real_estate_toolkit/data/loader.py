import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """
        Load data from a CSV file into a list of dictionaries using pandas.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a row of the CSV.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            df: pd.DataFrame = pd.read_csv(self.data_path, na_values=["NA"])  # Treat "NA" as missing
            # Convert the DataFrame to a list of dictionaries
            data = df.to_dict(orient="records")
            return data

        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    def validate_columns(self, required_columns: List[str]) -> bool:
        """
        Validate that all required columns are present in the dataset.

        Args:
            required_columns (List[str]): A list of required column names.

        Returns:
            bool: True if all required columns are present, otherwise False.
        """
        try:
            # Load the data to check for columns
            df: pd.DataFrame = pd.read_csv(self.data_path, na_values=["NA"])
            # Get the list of columns in the DataFrame
            existing_columns = df.columns.tolist()
            # Check if all required columns are present
            return all(column in existing_columns for column in required_columns)

        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
    











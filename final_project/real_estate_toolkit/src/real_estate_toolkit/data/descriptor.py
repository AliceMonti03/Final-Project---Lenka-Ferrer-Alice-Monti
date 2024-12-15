from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Any
import pandas as pd
import numpy as np

@dataclass
class Descriptor:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        df = pd.DataFrame(self.data)  # Create DataFrame from the data
        available_columns = df.columns.tolist()

        # If columns="all", process all columns
        if columns == "all":
            columns_to_process = available_columns
        else:
            invalid_columns = [col for col in columns if col not in available_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns specified: {invalid_columns}")
            columns_to_process = columns

        none_ratios = {}
        total_rows = len(df)  # Total number of rows in the DataFrame
        for col in columns_to_process:
            none_count = df[col].isna().sum()  # Count missing (NaN) values
            ratio = none_count / total_rows if total_rows > 0 else 0.0  # Calculate ratio
            none_ratios[col] = ratio

        return none_ratios

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables. Omit None values."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        df = pd.DataFrame(self.data)  # Create DataFrame from the data
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if columns == "all":
            columns_to_process = numeric_columns
        else:
            invalid_columns = [col for col in columns if col not in numeric_columns]
            if invalid_columns:
                raise ValueError(f"Invalid or non-numeric columns specified: {invalid_columns}")
            columns_to_process = columns

        averages = {}
        for col in columns_to_process:
            averages[col] = df[col].mean(skipna=True)  # Calculate average while ignoring None values

        return averages

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables. Omit None values."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        df = pd.DataFrame(self.data)  # Create DataFrame from the data
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if columns == "all":
            columns_to_process = numeric_columns
        else:
            invalid_columns = [col for col in columns if col not in numeric_columns]
            if invalid_columns:
                raise ValueError(f"Invalid or non-numeric columns specified: {invalid_columns}")
            columns_to_process = columns

        medians = {}
        for col in columns_to_process:
            medians[col] = df[col].median(skipna=True)  # Calculate median while ignoring None values

        return medians

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables. Omit None values.

        Args:
            columns (List[str] or str): A list of column names to compute percentiles for,
                                        or "all" to compute for all numeric columns.
            percentile (int): The percentile to compute (default is 50, i.e., the median).

        Returns:
            Dict[str, float]: A dictionary where keys are numeric column names and values are their computed percentiles.

        Raises:
            ValueError: If specified columns are invalid or not numeric, or if percentile is not in range [0, 100].
        """
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")

        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        df = pd.DataFrame(self.data)  # Create DataFrame from the data
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        # If columns="all", process all numeric columns
        if columns == "all":
            columns_to_process = numeric_columns
        else:
            invalid_columns = [col for col in columns if col not in numeric_columns]
            if invalid_columns:
                raise ValueError(f"Invalid or non-numeric columns specified: {invalid_columns}")
            columns_to_process = columns

        # Compute the percentile for each column, ignoring None values
        percentiles = {}
        for col in columns_to_process:
            percentiles[col] = df[col].dropna().quantile(percentile / 100.0)  # Compute the percentile

        return percentiles
    
    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """
        Compute the mode for variables. Omit None values.

        Args:
            columns (List[str] or str): A list of column names to compute modes for, or "all" to compute for all columns.

        Returns:
            Dict[str, Union[Tuple[str, float], Tuple[str, str]]]: A dictionary where keys are column names and values
            are tuples of the variable type and the mode.

        Raises:
            ValueError: If specified columns are invalid.
        """
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        # Create DataFrame from the data
        df = pd.DataFrame(self.data)

        # Get all available columns
        available_columns = df.columns.tolist()

        # If columns="all", process all columns
        if columns == "all":
            columns_to_process = available_columns
        else:
            # Validate specified columns
            invalid_columns = [col for col in columns if col not in available_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns specified: {invalid_columns}")
            columns_to_process = columns

        # Compute the type and mode for each column
        result = {}
        for col in columns_to_process:
            # Get the column data type
            dtype = str(df[col].dtype)
            
            # Calculate the mode, excluding None values
            mode_series = df[col].dropna().mode()
            mode = mode_series.iloc[0] if not mode_series.empty else None

            # Format the result based on type
            if pd.api.types.is_numeric_dtype(df[col]):
                result[col] = (dtype, mode if mode is not None else float("nan"))
            else:
                result[col] = (dtype, mode if mode is not None else "None")

        return result
    

# 3.1.2.2 Implement descriptor class using NumPy
@dataclass
class DescriptorNumpy:
    """Class for cleaning real estate data using NumPy."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None (or NaN) values per column."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        # Convert data to a structured NumPy array
        dtype = [(col, 'O') for col in self.data[0].keys()]  # Flexible dtype for mixed types
        np_data = np.array([tuple(row.values()) for row in self.data], dtype=dtype)

        available_columns = list(np_data.dtype.names)

        # Determine columns to process
        if columns == "all":
            columns_to_process = available_columns
        else:
            invalid_columns = [col for col in columns if col not in available_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns specified: {invalid_columns}")
            columns_to_process = columns

        none_ratios = {}
        total_rows = np_data.shape[0]  # Total number of rows

        for col in columns_to_process:
            col_data = np.array(np_data[col])
            none_count = np.sum(pd.isnull(col_data))  # Use pandas for detecting NaN
            ratio = none_count / total_rows if total_rows > 0 else 0.0
            none_ratios[col] = ratio

        return none_ratios

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables. Omit None values."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        dtype = [(col, 'O') for col in self.data[0].keys()]
        np_data = np.array([tuple(row.values()) for row in self.data], dtype=dtype)

        numeric_columns = [
            col for col in np_data.dtype.names
            if np.issubdtype(np.array(np_data[col]).dtype, np.number)
        ]

        if columns == "all":
            columns_to_process = numeric_columns
        else:
            invalid_columns = [col for col in columns if col not in numeric_columns]
            if invalid_columns:
                raise ValueError(f"Invalid or non-numeric columns specified: {invalid_columns}")
            columns_to_process = columns

        averages = {}
        for col in columns_to_process:
            col_data = np.array(np_data[col], dtype=float)
            averages[col] = np.nanmean(col_data)  # Ignore NaN values

        return averages

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables. Omit None values."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        dtype = [(col, 'O') for col in self.data[0].keys()]
        np_data = np.array([tuple(row.values()) for row in self.data], dtype=dtype)

        numeric_columns = [
            col for col in np_data.dtype.names
            if np.issubdtype(np.array(np_data[col]).dtype, np.number)
        ]

        if columns == "all":
            columns_to_process = numeric_columns
        else:
            invalid_columns = [col for col in columns if col not in numeric_columns]
            if invalid_columns:
                raise ValueError(f"Invalid or non-numeric columns specified: {invalid_columns}")
            columns_to_process = columns

        medians = {}
        for col in columns_to_process:
            col_data = np.array(np_data[col], dtype=float)
            medians[col] = np.nanmedian(col_data)  # Ignore NaN values

        return medians

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables. Omit None values."""
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")

        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        dtype = [(col, 'O') for col in self.data[0].keys()]
        np_data = np.array([tuple(row.values()) for row in self.data], dtype=dtype)

        numeric_columns = [
            col for col in np_data.dtype.names
            if np.issubdtype(np.array(np_data[col]).dtype, np.number)
        ]

        if columns == "all":
            columns_to_process = numeric_columns
        else:
            invalid_columns = [col for col in columns if col not in numeric_columns]
            if invalid_columns:
                raise ValueError(f"Invalid or non-numeric columns specified: {invalid_columns}")
            columns_to_process = columns

        percentiles = {}
        for col in columns_to_process:
            col_data = np.array(np_data[col], dtype=float)
            percentiles[col] = np.nanpercentile(col_data, percentile)  # Compute percentile ignoring NaN

        return percentiles

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """Compute the mode for variables. Omit None values."""
        if not self.data:
            raise ValueError("Data is empty or has not been loaded.")

        dtype = [(col, 'O') for col in self.data[0].keys()]
        np_data = np.array([tuple(row.values()) for row in self.data], dtype=dtype)

        available_columns = list(np_data.dtype.names)

        if columns == "all":
            columns_to_process = available_columns
        else:
            invalid_columns = [col for col in columns if col not in available_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns specified: {invalid_columns}")
            columns_to_process = columns

        result = {}
        for col in columns_to_process:
            col_data = np.array(np_data[col])
            dtype = str(col_data.dtype)

            # Compute the mode ignoring NaN
            col_data_clean = col_data[~pd.isnull(col_data)]
            if col_data_clean.size == 0:
                mode = None
            else:
                mode = np.unique(col_data_clean)[0]

            if np.issubdtype(col_data.dtype, np.number):
                result[col] = (dtype, mode if mode is not None else float("nan"))
            else:
                result[col] = (dtype, mode if mode is not None else "None")

        return result
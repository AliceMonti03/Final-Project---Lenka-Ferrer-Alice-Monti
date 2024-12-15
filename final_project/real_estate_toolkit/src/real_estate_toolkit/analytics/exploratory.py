from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import polars as pl
import plotly.express as px
import os
import plotly.graph_objects as go


class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        """
        try:
            # Load the dataset into a Polars DataFrame with specified options
            self.real_state_data = pl.read_csv(
                data_path,
                null_values=["NA"],  # Specify NA as null
            )
            print(f"Data loaded successfully from {data_path}.")
            
            # Initialize the clean data attribute as None (to be processed later)
            self.real_state_clean_data = None

        except Exception as e:
            print(f"Error loading data from {data_path}: {str(e)}")
   
    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning using Polars.
        """
        print("Starting data cleaning...")

        # Handle missing values
        total_rows = self.real_state_data.height

        # Separate numeric and categorical columns
        numeric_columns = [col for col in self.real_state_data.columns if 
                        self.real_state_data[col].dtype in [pl.Float64, pl.Int64]]
        categorical_columns = [col for col in self.real_state_data.columns if
                            self.real_state_data[col].dtype == pl.Utf8]

        # Fill missing values in numeric columns with the mean
        self.real_state_data = self.real_state_data.with_columns([
            pl.when(pl.col(col).is_null()).then(self.real_state_data[col].mean()).otherwise(pl.col(col)).alias(col)
            for col in numeric_columns
        ])

        # Fill missing values in categorical columns with the mode
        self.real_state_data = self.real_state_data.with_columns([
            pl.when(pl.col(col).is_null()).then(pl.lit(self.real_state_data[col].mode().first())).otherwise(pl.col(col)).alias(col)
            for col in categorical_columns
        ])

        # Assign cleaned data to the attribute
        self.real_state_clean_data = self.real_state_data

        print(f"Data cleaned successfully. Remaining rows: {self.real_state_clean_data.height}")

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        
        Tasks to implement:
        1. Compute basic price statistics and generate another data frame called price_statistics:
            - Mean
            - Median
            - Standard deviation
            - Minimum and maximum prices
        2. Create an interactive histogram of sale prices using Plotly.
        
        Returns:
            - Statistical insights dataframe
            - Save Plotly figures for price distribution in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Cleaned data is not available. Please run clean_data() first.")

        # Step 1: Compute basic price statistics
        price_statistics = self.real_state_clean_data.select(
            [
                pl.col("SalePrice").mean().alias("mean_price"),
                pl.col("SalePrice").median().alias("median_price"),
                pl.col("SalePrice").std().alias("std_dev_price"),
                pl.col("SalePrice").min().alias("min_price"),
                pl.col("SalePrice").max().alias("max_price"),
            ]
        )

        print("Price Statistics:", price_statistics)  # Debug output to check statistics
        
        # Step 2: Create an interactive histogram of sale prices using Plotly
        fig = px.histogram(
            self.real_state_clean_data.to_pandas(),  # Convert the Polars DataFrame to Pandas
            x="SalePrice",
            title="Histogram of Sale Prices",
            labels={"SalePrice": "Sale Price"},
            color_discrete_sequence=["blue"],  # Customize the color if needed
            marginal="rug"  # Optional: Add a rug plot to show individual observations
        )

        # Define the output folder and ensure it exists
        output_folder = "real_estate_toolkit/src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        # Save the figure as an HTML file
        histogram_file_path = os.path.join(output_folder, "price_distribution_histogram.html")
        fig.write_html(histogram_file_path)
        print(f"Histogram saved to {histogram_file_path}")

        return price_statistics  # Return the price statistics DataFrame
    
    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        
        Tasks to implement:
        1. Group data by neighborhood
        2. Calculate price statistics for each neighborhood
        3. Create Plotly boxplot with:
            - Median prices
            - Price spread
            - Outliers
        
        Returns:
            - Return neighborhood statistics dataframe
            - Save Plotly figures for neighborhood price comparison in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Cleaned data is not available. Please run clean_data() first.")

        # Step 1: Group data by neighborhood and compute statistics
        neighborhood_stats = self.real_state_clean_data.group_by("Neighborhood").agg(
            [
                pl.col("SalePrice").mean().alias("mean_price"),
                pl.col("SalePrice").median().alias("median_price"),
                pl.col("SalePrice").std().alias("std_dev_price"),
                pl.col("SalePrice").min().alias("min_price"),
                pl.col("SalePrice").max().alias("max_price"),
            ]
        )

        print("Neighborhood Price Statistics:", neighborhood_stats)  # Debug output

        # Step 2: Create a Plotly boxplot for sale prices by neighborhood
        fig = px.box(
            self.real_state_clean_data.to_pandas(),
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison",
            labels={"Neighborhood": "Neighborhood", "SalePrice": "Sale Price"},
            color="Neighborhood",  # Assign a unique color to each neighborhood
        )

        # Customize layout to improve readability
        fig.update_layout(
            xaxis_title="Neighborhood",
            yaxis_title="Sale Price",
            xaxis=dict(tickangle=45),  # Rotate neighborhood labels for better visibility
            showlegend=False  # Hide legend to avoid clutter
        )

        # Define the output folder and ensure it exists
        output_folder = "real_estate_toolkit/src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        # Save the figure as an HTML file
        boxplot_file_path = os.path.join(output_folder, "neighborhood_price_comparison_boxplot.html")
        fig.write_html(boxplot_file_path)
        print(f"Boxplot saved to {boxplot_file_path}")

        return neighborhood_stats  # Return the neighborhood statistics DataFrame
    
    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for variables input.
        
        Tasks to implement:
        1. Pass a list of numerical variables
        2. Compute correlation matrix and plot it
        
        Args:
            variables (List[str]): List of variables to correlate
        
        Returns:
            Save Plotly figures for correlation heatmap in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Cleaned data is not available. Please run clean_data() first.")

        # Ensure all variables exist in the cleaned dataset
        for var in variables:
            if var not in self.real_state_clean_data.columns:
                raise ValueError(f"Variable '{var}' not found in the dataset.")

        # Select the specified columns
        selected_data = self.real_state_clean_data.select(variables)

        # Compute correlation matrix using Polars
        correlation_matrix = selected_data.to_pandas().corr()

        # Create a Plotly heatmap
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=variables,
            y=variables,
            color_continuous_scale="RdBu",
            title="Correlation Heatmap",
        )

        # Define the output folder and ensure it exists
        output_folder = "real_estate_toolkit/src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        # Save the figure as an HTML file
        heatmap_file_path = os.path.join(output_folder, "correlation_heatmap.html")
        fig.write_html(heatmap_file_path)
        print(f"Correlation heatmap saved to {heatmap_file_path}")
    
    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        
        Scatter plots to create:
        1. House price vs. Total square footage
        2. Sale price vs. Year built
        3. Overall quality vs. Sale price
        
        Tasks to implement:
        - Use Plotly Express for creating scatter plots
        - Add trend lines
        - Include hover information
        - Color-code points based on a categorical variable
        - Save them in src/real_estate_toolkit/analytics/outputs/ folder.
        
        Returns:
            Dictionary of Plotly Figure objects for different scatter plots. 
        """
        if self.real_state_clean_data is None:
            raise ValueError("Cleaned data is not available. Please run clean_data() first.")

        # Create a dictionary to store the scatter plot figures
        scatter_plots = {}

        # 1. House price vs. Total square footage
        fig1 = px.scatter(
            self.real_state_clean_data.to_pandas(),
            x="GrLivArea",  # Total square footage
            y="SalePrice",
            title="House Price vs. Total Square Footage",
            labels={"GrLivArea": "Total Square Footage", "SalePrice": "Sale Price"},
            color="Neighborhood",  # Color-code by Neighborhood (or any other categorical variable)
            trendline="ols",  # Add trendline (Ordinary Least Squares)
            hover_data=["Neighborhood", "OverallQual"],  # Additional hover info
        )
        scatter_plots["house_price_vs_sqft"] = fig1

        # 2. Sale price vs. Year built
        fig2 = px.scatter(
            self.real_state_clean_data.to_pandas(),
            x="YearBuilt",  # Year built
            y="SalePrice",
            title="Sale Price vs. Year Built",
            labels={"YearBuilt": "Year Built", "SalePrice": "Sale Price"},
            color="Neighborhood",  # Color-code by Neighborhood (or any other categorical variable)
            trendline="ols",  # Add trendline (Ordinary Least Squares)
            hover_data=["Neighborhood", "OverallQual"],  # Additional hover info
        )
        scatter_plots["sale_price_vs_year_built"] = fig2

        # 3. Overall quality vs. Sale price
        fig3 = px.scatter(
            self.real_state_clean_data.to_pandas(),
            x="OverallQual",  # Overall quality
            y="SalePrice",
            title="Overall Quality vs. Sale Price",
            labels={"OverallQual": "Overall Quality", "SalePrice": "Sale Price"},
            color="Neighborhood",  # Color-code by Neighborhood (or any other categorical variable)
            trendline="ols",  # Add trendline (Ordinary Least Squares)
            hover_data=["Neighborhood", "GrLivArea"],  # Additional hover info
        )
        scatter_plots["overall_quality_vs_sale_price"] = fig3

        # Save the figures to HTML files
        output_folder = "real_estate_toolkit/src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        for plot_name, fig in scatter_plots.items():
            file_path = os.path.join(output_folder, f"{plot_name}.html")
            fig.write_html(file_path)
            print(f"Scatter plot saved to {file_path}")

        return scatter_plots  # Return the dictionary of scatter plot figures
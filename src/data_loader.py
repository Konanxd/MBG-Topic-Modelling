import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data Loader class for handling tweet data loading and basic validation.
    
    Why this class?
    - Separates data loading logic from other components (Single Responsibility Principle)
    - Provides reusable methods for different data sources
    - Includes validation to catch data quality issues early
    """

    def __init__(self, file_path: str, text_column: str = "text"):
        """
        Initialize DataLoader with file path and text column name.
        
        Args:
            file_path: Path to the CSV file containing tweets
            text_column: Name of the column containing tweet text
        """
        self.file_path = file_path
        self.text_column = text_column
        self.data = None
     
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded {len(self.data)} records!")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self) -> bool:
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.text_column not in self.data.columns:
            raise ValueError(f"Column '{self.text_column}' not found in data")
        
        null_count = self.data[self.text_column].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in text column")
        
        empty_count = (self.data[self.text_column] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty values in text column")
        
        logger.info("Data validation complete")
        return True
    
    def get_basic_stats(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        texts = self.data[self.text_column].dropna()

        stats = {
            'total_records' : len(self.data),
            'non_null_texts' : len(texts),
            'avg_text_length' : texts.str.len().mean(),
            'median_text_length' : texts.str.len().median(),
            'min_text_length' : texts.str.len().min(),
            'max_text_length' : texts.str.len().max(),
        }

        logger.info(f"Dataset statistics: {stats}")
        return stats
    
    def sample_data(self, n: int = 5) -> pd.DataFrame:
        
        """
        Get a random sample of the data.
        
        Args:
            n: Number of samples to return
            
        Returns:
            DataFrame with n random samples
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        return self.data.sample(n=min(n, len(self.data)))
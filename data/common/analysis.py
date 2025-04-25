# data/common/analysis.py
import pandas as pd
import numpy as np
from typing import Dict, Union

class Analysis:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a DataFrame containing account performance data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Must contain:
            - account_id: Unique account identifier
            - observation_date: Date of observation
            - delinquency_status: Days delinquent (0=current, 30=30 days late, etc.)
        """
        required_columns = {'account_id', 'observation_date', 'delinquency_status'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        self.data = data.sort_values(['account_id', 'observation_date'])
        
    def roll_rate_analysis(self, 
                          period: str = 'M',
                          lookback_periods: int = 12) -> pd.DataFrame:
        """
        Perform roll rate analysis across specified time periods
        
        Parameters:
        -----------
        period : str
            Time period for analysis ('M' for monthly, 'Q' for quarterly)
        lookback_periods : int
            Number of periods to analyze
            
        Returns:
        --------
        pd.DataFrame
            Roll rate matrix with transition probabilities
        """
        # Create period reference
        self.data['period'] = self.data['observation_date'].dt.to_period(period)
        
        # Get unique periods sorted
        periods = self.data['period'].dropna().sort_values().unique()
        if len(periods) < 2:
            raise ValueError("Insufficient data periods for roll rate analysis")
            
        # Create lagged status
        self.data['prev_status'] = self.data.groupby('account_id')['delinquency_status'].shift(1)
        self.data['next_status'] = self.data.groupby('account_id')['delinquency_status'].shift(-1)
        
        # Filter relevant periods
        analysis_data = self.data.dropna(subset=['prev_status', 'next_status'])
        
        # Create status buckets
        status_bins = [-1, 0, 30, 60, 90, 180, np.inf]
        analysis_data['current_bucket'] = pd.cut(analysis_data['prev_status'], bins=status_bins)
        analysis_data['next_bucket'] = pd.cut(analysis_data['next_status'], bins=status_bins)
        
        # Create roll rate matrix
        roll_matrix = pd.crosstab(
            index=analysis_data['current_bucket'],
            columns=analysis_data['next_bucket'],
            normalize='index'
        ).round(4) * 100
        
        # Calculate roll rates
        roll_matrix['total'] = roll_matrix.sum(axis=1)
        roll_matrix = roll_matrix.sort_index(ascending=False)
        
        return roll_matrix

# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='M')
    accounts = pd.DataFrame({
        'account_id': np.repeat(np.arange(1, 101), len(dates)),
        'observation_date': np.tile(dates, 100),
        'delinquency_status': np.random.choice([0, 30, 60, 90, 180], size=100*len(dates))
    })
    
    # Add some progression logic
    accounts['delinquency_status'] = accounts.groupby('account_id')['delinquency_status'].cummax()
    
    # Initialize analysis
    analyzer = Analysis(accounts)
    
    # Perform roll rate analysis
    roll_rates = analyzer.roll_rate_analysis(period='M', lookback_periods=6)
    
    print("Roll Rate Analysis Matrix:")
    print(roll_rates)
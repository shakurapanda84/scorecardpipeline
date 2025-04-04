import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def analyze_null_variables(df, ratio=0.1, plot=True, detailed=True):
        """
        Analyze and summarize variables with null values percentage greater than specified ratio
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input DataFrame
        ratio : float, default=0.1
            Threshold ratio for null values (0 to 1)
        plot : bool, default=True
            Whether to plot the null value distribution
        detailed : bool, default=True
            Whether to return detailed statistics
            
        Returns:
        --------
        dict : Dictionary containing analysis results
        """
        
        # Calculate null percentages
        null_percentages = df.isnull().mean().round(4) * 100
        high_null_vars = null_percentages[null_percentages > ratio * 100]
        
        # Basic summary
        summary = {
            'total_variables': len(df.columns),
            'high_null_variables': len(high_null_vars),
            'high_null_percentage': (len(high_null_vars) / len(df.columns) * 100).round(2)
        }
        
        # Detailed analysis for high-null variables
        if len(high_null_vars) > 0:
            detailed_analysis = pd.DataFrame({
                'null_percentage': high_null_vars,
                'non_null_count': df[high_null_vars.index].count(),
                'dtype': df[high_null_vars.index].dtypes,
                'unique_values': df[high_null_vars.index].nunique(),
            }).sort_values('null_percentage', ascending=False)
            
            # Add memory usage if detailed
            if detailed:
                detailed_analysis['memory_usage_mb'] = df[high_null_vars.index].memory_usage(deep=True) / 1024 / 1024
                
                # Add sample non-null values
                sample_values = {}
                for col in high_null_vars.index:
                    sample = df[col].dropna().sample(min(5, df[col].count())).tolist()
                    sample_values[col] = sample
                detailed_analysis['sample_values'] = detailed_analysis.index.map(sample_values)
        
        # Plotting
        if plot and len(high_null_vars) > 0:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=high_null_vars.index, y=high_null_vars.values)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Variables with >{ratio*100}% Null Values')
            plt.ylabel('Null Percentage')
            plt.tight_layout()
            plt.show()
        
        # Generate recommendations
        recommendations = []
        for col, null_pct in high_null_vars.items():
            if null_pct > 90:
                recommendations.append(f"Consider dropping {col} ({null_pct:.1f}% missing)")
            elif df[col].dtype in ['int64', 'float64']:
                recommendations.append(f"Consider imputing {col} with median/mean")
            else:
                recommendations.append(f"Consider imputing {col} with mode or creating missing category")
        
        result = {
            'summary': summary,
            'high_null_variables': high_null_vars.to_dict(),
            'recommendations': recommendations
        }
        
        if detailed and len(high_null_vars) > 0:
            result['detailed_analysis'] = detailed_analysis
        
        return result 
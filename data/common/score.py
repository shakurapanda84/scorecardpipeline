import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Scorecard:
    def __init__(self, pdo, base_odds, min_score, max_score):
        """
        Initialize the Scorecard with the given parameters.

        Parameters:
        -----------
        pdo : int
            Points to Double Odds.
        base_odds : float
            Base odds for the scorecard.
        min_score : int
            Minimum score for the scorecard.
        max_score : int
            Maximum score for the scorecard.
        """
        self.pdo = pdo
        self.base_odds = base_odds
        self.min_score = min_score
        self.max_score = max_score
        self.factor = self.pdo / np.log(2)
        self.offset = self.min_score - self.factor * np.log(self.base_odds)

    def createScorecard(self, probabilities):
        """
        Calculate the credit scores for the given probabilities.

        Parameters:
        -----------
        probabilities : list of float
            List of predicted probabilities.

        Returns:
        --------
        list of float : Calculated scores.
        """
        scores = []
        for probability in probabilities:
            odds = (1 - probability) / probability
            score = self.offset + self.factor * np.log(odds)
            scores.append(score)
        return scores


    def score_breakdown(df, score_col, bad_col='bad', pdo=20):
        """
        Create a score breakdown analysis for a given score column
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing scores and bad indicator
        score_col : str
            Name of the score column to analyze
        bad_col : str, default 'bad'
            Name of the bad indicator column (1=bad, 0=good)
        pdo : int, default 20
            Points to Double Odds - used as bin size for score ranges
            
        Returns:
        --------
        pd.DataFrame
            Score breakdown with columns:
            - Score Range
            - Total Count
            - Bad Count
            - Bad Rate
        """
        # Validate inputs
        if score_col not in df.columns:
            raise ValueError(f"Column '{score_col}' not found in dataframe")
        if bad_col not in df.columns:
            raise ValueError(f"Bad indicator column '{bad_col}' not found")
        if pdo <= 0:
            raise ValueError("PDO must be a positive integer")
        
        # Filter out missing scores
        clean_df = df[[score_col, bad_col]].dropna(subset=[score_col])
        
        # Create bins: [-inf, 500), [500,520), [520,540), etc.
        max_score = np.ceil(clean_df[score_col].max() / pdo) * pdo
        # Generate unique sorted bins
        lower_bins = [-np.inf, 500]
        upper_bins = list(np.arange(500 + pdo, max_score + pdo + 1e-9, pdo))  # Add small epsilon
        
        # Create bins with unique values using pandas' IntervalIndex
        all_bins = lower_bins + upper_bins
        bins = pd.unique(all_bins)  # Properly handle numeric precision
        bins.sort()
        
        # Create labels for bins
        labels = []
        for i in range(len(bins)-1):
            if i == 0 and np.isneginf(bins[i]):
                labels.append(f"<{bins[i+1]}")
            else:
                labels.append(f"{int(bins[i])}-{int(bins[i+1])}")
        
        # Create bins and calculate metrics
        clean_df['Score Range'] = pd.cut(clean_df[score_col], bins=bins, 
                                        labels=labels, 
                                        right=False,
                                        include_lowest=True)
        
        # Group by score range
        breakdown = clean_df.groupby('Score Range', observed=True).agg(
            Total_Count=(bad_col, 'count'),
            Bad_Count=(bad_col, lambda x: x.clip(lower=0).sum())
        ).reset_index()
        
        # Calculate bad rate
        breakdown['Bad_Rate'] = breakdown['Bad_Count'] / breakdown['Total_Count']
        
        # Handle any empty bins
        full_range = pd.DataFrame({'Score Range': labels})
        breakdown = full_range.merge(breakdown, how='left').fillna(0)
        
        # Format columns
        breakdown.columns = ['Score Range', 'Total Count', 'Bad Count', 'Bad Rate']
        
        return breakdown


def plot_bad_ratio(breakdown_df, figsize=(12, 6)):
    """
    Create a bar plot showing the Bad Rate distribution across score ranges
    
    Parameters:
    -----------
    breakdown_df : pd.DataFrame
        DataFrame from score_breakdown() function
    figsize : tuple, default (12,6)
        Size of the figure (width, height)
    """
    plt.figure(figsize=figsize)
    ax = breakdown_df.plot.bar(x='Score Range', y='Bad Rate', legend=False)
    
    # Formatting
    ax.set_title('Bad Rate Distribution by Score Range', fontsize=14)
    ax.set_xlabel('Score Range', fontsize=12)
    ax.set_ylabel('Bad Rate (%)', fontsize=12)
    ax.set_xticklabels(breakdown_df['Score Range'], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initialize the Scorecard
    scorecard = Scorecard(pdo=30, base_odds=1, min_score=200, max_score=800)

    # Example probabilities
    probabilities = [0.1, 0.2, 0.5, 0.7, 0.9]

    # Calculate scores
    scores = scorecard.createScorecard(probabilities)

    # Display the results
    for prob, score in zip(probabilities, scores):
        print(f"Probability: {prob:.2f}, Score: {score:.2f}") 
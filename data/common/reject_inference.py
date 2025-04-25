import numpy as np
import pandas as pd

class RejectInference:
    def __init__(self, accepted_data, rejected_data, target_column):
        """
        Initialize the RejectInference with accepted and rejected datasets.

        Parameters:
        -----------
        accepted_data : pd.DataFrame
            DataFrame containing accepted applicants with known outcomes.
        rejected_data : pd.DataFrame
            DataFrame containing rejected applicants.
        target_column : str
            The column name for the target variable (e.g., 'default').
        """
        self.accepted_data = accepted_data
        self.rejected_data = rejected_data
        self.target_column = target_column

    def augmentation(self):
        """
        Augmentation method: Assigns a default status to all rejected applicants.
        """
        self.rejected_data[self.target_column] = 1
        return self.rejected_data

    def fuzzy_augmentation(self, default_probability=0.5):
        """
        Fuzzy Augmentation method: Assigns a default status based on a probability.

        Parameters:
        -----------
        default_probability : float
            Probability of assigning a default status to rejected applicants.
        """
        self.rejected_data[self.target_column] = np.random.binomial(1, default_probability, len(self.rejected_data))
        return self.rejected_data

    def parceling(self):
        """
        Parceling method: Distributes rejected applicants into good and bad categories.
        """
        default_rate = self.accepted_data[self.target_column].mean()
        self.rejected_data[self.target_column] = np.random.binomial(1, default_rate, len(self.rejected_data))
        return self.rejected_data

    def extrapolation(self, model):
        """
        Extrapolation method: Uses a model to predict the likelihood of default.

        Parameters:
        -----------
        model : sklearn-like model
            A trained model with a predict_proba method.
        """
        probabilities = model.predict_proba(self.rejected_data.drop(columns=[self.target_column]))[:, 1]
        self.rejected_data[self.target_column] = np.random.binomial(1, probabilities)
        return self.rejected_data

    def masked_sampling(self, predict_proba, n_sample=1000, ratio=1, bad_threshold=0.8, good_threshold=0.2):
        """
        Samples rejected data based on predicted probabilities and thresholds.

        Parameters:
        -----------
        predict_proba : array-like
            Predicted probabilities for rejected samples
        n_sample : int
            Number of samples to draw from bad cases
        ratio : float
            Ratio of good to bad samples
        bad_threshold : float
            Upper threshold for identifying bad cases (prob > bad_threshold)
        good_threshold : float
            Lower threshold for identifying good cases (prob < good_threshold)
        """
        import random
        from sklearn.utils import shuffle

        # Add predicted probabilities
        self.rejected_data['user_prob'] = predict_proba
        
        # Filter samples based on thresholds
        mask = ((self.rejected_data['user_prob'] > bad_threshold) & 
                (self.rejected_data['user_prob'] < 1)) | \
               ((self.rejected_data['user_prob'] < good_threshold) & 
                (self.rejected_data['user_prob'] > 0))
        filtered_data = self.rejected_data[mask].copy()
        
        print("The filtered data is ")
        print(filtered_data)
        # Assign binary labels based on probability
        filtered_data[self.target_column] = (filtered_data['user_prob'] > 0.5).astype(float)
        
        print(filtered_data[self.target_column].value_counts())
        # Sample from both classes
        n = random.randint(1, 10)
        bad_samples = filtered_data[filtered_data[self.target_column] == 1].sample(n=n_sample, random_state=n)
        good_samples = filtered_data[filtered_data[self.target_column] == 0].sample(n=int(ratio * n_sample), random_state=n)
        
        # Combine and shuffle
        combined_samples = shuffle(pd.concat([good_samples, bad_samples]))
        return combined_samples

# Example usage
if __name__ == "__main__":
    # Sample data
    accepted_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'default': np.random.binomial(1, 0.2, 100)
    })
    rejected_data = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50)
    })

    # Initialize RejectInference
    reject_inference = RejectInference(accepted_data, rejected_data, 'default')

    # Generate some predicted probabilities for rejected data
    from sklearn.linear_model import LogisticRegression
    
    # Train a model on accepted data
    model = LogisticRegression()
    y_accepted = accepted_data['default']
    X_accepted = accepted_data.drop(columns=['default'])
  
    model.fit(X_accepted, y_accepted)
    
    # Generate predictions for rejected data
    predicted_probs = model.predict_proba(rejected_data)[:, 1]
    print(predicted_probs)
    # Apply masked sampling
    masked_samples = reject_inference.masked_sampling(
        predict_proba=predicted_probs,
        n_sample=20,  # smaller sample size for demonstration
        ratio=1.5,    # will sample 30 good cases (1.5 * 20)
        bad_threshold=0.8,
        good_threshold=0.2
    )
    
    print("\nSampled Data Shape:", masked_samples.shape)
    print("\nClass Distribution:")
    print(masked_samples['default'].value_counts())
    print("\nFirst few samples:")
    print(masked_samples.head()) 
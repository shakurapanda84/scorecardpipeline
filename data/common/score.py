import numpy as np

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

# Example usage
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
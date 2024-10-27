import numpy as np
from bandits.algorithm.ContextualBandit import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling algorithm implementation for non-contextual bandits.

    Attributes:
        alpha (np.ndarray): Success counts for Beta distribution.
        beta (np.ndarray): Failure counts for Beta distribution.
    """

    def __init__(self, n_arms: int):
        """
        Initialize the Thompson Sampling algorithm.

        Args:
            n_arms (int): Number of arms.
        """
        super().__init__(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self, context: np.ndarray = None) -> int:
        """
        Select an arm using Thompson Sampling.

        Args:
            context (np.ndarray, optional): Contextual feature vector (ignored).

        Returns:
            int: The index of the selected arm.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, context: np.ndarray = None, reward: float = None):
        """
        Update the alpha and beta parameters for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray, optional): Contextual feature vector (ignored).
            reward (float): Observed reward (assumed to be 0 or 1).
        """
        # Assuming reward is between 0 and 1
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
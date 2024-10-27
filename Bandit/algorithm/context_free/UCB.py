import numpy as np
from bandits.algorithm.ContextualBandit import BanditAlgorithm


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm implementation for non-contextual bandits.

    Attributes:
        counts (np.ndarray): Number of times each arm has been selected.
        values (np.ndarray): Average reward for each arm.
    """

    def __init__(self, n_arms: int):
        """
        Initialize the UCB algorithm.

        Args:
            n_arms (int): Number of arms.
        """
        super().__init__(n_arms)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self, context: np.ndarray = None) -> int:
        """
        Select an arm using the UCB algorithm.

        Args:
            context (np.ndarray, optional): Contextual feature vector (ignored).

        Returns:
            int: The index of the selected arm.
        """
        if self.total_counts < self.n_arms:
            return self.total_counts  # Play each arm once to initialize
        else:
            ucb_values = self.values + np.sqrt(
                (2 * np.log(self.total_counts)) / self.counts
            )
            return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray = None, reward: float = None):
        """
        Update the estimated values for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray, optional): Contextual feature vector (ignored).
            reward (float): Observed reward.
        """
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Update the average reward
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
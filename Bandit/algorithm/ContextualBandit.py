import numpy as np


class BanditAlgorithm:
    """
    Base class for contextual bandit algorithms.
    """

    def __init__(self, arms, n_features: int = None):
        """
        Initialize the contextual bandit algorithm.
        """
        self.arms = arms
        self.n_arms = len(arms)
        self.n_features = n_features
        self.arms_mapping = self._preprocess_data()

    def select_arm(self, context: np.ndarray, arm_id: int) -> int:
        """
        Select an arm based on the provided context.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(self, arm: int, context: np.ndarray, reward: np.ndarray):
        """
        Update the algorithm's parameters based on the observed reward.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _preprocess_data(self):
        """
        Map the index of the arm to product_id
        """
        mapping = dict(zip(range(self.n_arms), np.unique(self.arms).tolist()))
        return mapping

    def _get_key(self, value):
        """
        Get the index of the arm based on the provided product_id
        """
        temp = [k for k, v in self.arms_mapping.items() if v == value]
        if len(temp) == 0:
            key = None
        else:
            key = temp[0]
        return key

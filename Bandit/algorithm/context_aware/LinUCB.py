import numpy as np
from bandits.algorithm.ContextualBandit import BanditAlgorithm


class LinUCB(BanditAlgorithm):
    """
    LinUCB algorithm implementation.

    Attributes:
        alpha (float): Exploration parameter.
        A (list): List of A matrices for each arm.
        b (list): List of b vectors for each arm.
    """

    def __init__(self, arms, n_features: int, alpha: float = 1.0):
        """
        Initialize the LinUCB algorithm.
        """
        super().__init__(arms, n_features)
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(self.n_arms)]

    def select_arm(self, context: np.ndarray, arm_id: int) -> int:
        """
        Select an arm using the LinUCB algorithm.
        """
        context = context.reshape(-1, 1)
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p[arm] = (
                    theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)
            ).item()
        return int(self.arms_mapping[np.argmax(p)])

    def update(self, arm: int, context: np.ndarray, reward: np.ndarray):
        """
        Update the parameters of the selected arm.
        """
        context = context.reshape(-1, 1)
        arm = self._get_key(arm)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context

    def predict(self, arm, context: np.ndarray):
        arm = self._get_key(arm)
        if arm:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            score = (
                    theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)
            ).item()
        else:
            score = None
        return score

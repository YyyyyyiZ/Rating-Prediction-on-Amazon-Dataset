import numpy as np
from bandits.algorithm.ContextualBandit import BanditAlgorithm


class KernelUCB(BanditAlgorithm):
    """
    KernelUCB algorithm implementation.

    Attributes:
        alpha (float): Exploration parameter.
        contexts (list): List of observed contexts.
        rewards (list): List of observed rewards.
        kernel (callable): Kernel function.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: float = 1.0,
    ):
        """
        Initialize the KernelUCB algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of contextual features.
            alpha (float, optional): Exploration parameter.
            kernel (str, optional): Type of kernel ('rbf', 'linear').
            gamma (float, optional): Kernel coefficient for 'rbf'.
        """
        super().__init__(n_arms, n_features)
        self.alpha = alpha
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]
        if kernel == "rbf":
            self.kernel = lambda x, y: np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        elif kernel == "linear":
            self.kernel = lambda x, y: np.dot(x, y)
        else:
            raise ValueError("Unsupported kernel type")

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the KernelUCB algorithm.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.contexts[arm]:
                K = np.array([self.kernel(context, x) for x in self.contexts[arm]])
                K_inv = np.linalg.inv(np.outer(K, K) + np.eye(len(K)) * 1e-5)
                rewards = np.array(self.rewards[arm])
                mu = K @ K_inv @ rewards
                sigma = np.sqrt(self.kernel(context, context) - K @ K_inv @ K)
                p[arm] = mu + self.alpha * sigma
            else:
                p[arm] = self.alpha * np.sqrt(self.kernel(context, context))
        return int(np.argmax(p))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the observations for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
        """
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)
# Amazon Dataset Rating Prediction
## Dataset
For more details, refer to https://darel13712.github.io/rs_datasets/

## Matrix Factorization
### FunkSVD
### BiasSVD
### Probabilistic Matrix Factorization (PMF)
[Mnih, A., & Salakhutdinov, R. R. Probabilistic matrix factorization. NIPS 2008](https://papers.nips.cc/paper_files/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
### Bayesian Personalized Ranking (BPR)
[Rendle et. al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009](https://arxiv.org/pdf/1205.2618)

## Deep Neural Network

## Multi-Armed Bandit
### Existing Algorithms
Below are detailed descriptions of each algorithm, indicating whether they are contextual or non-contextual, how they work, and examples demonstrating their usage.
#### Context-free
##### Upper Confidence Bound (UCB)
* **Description**:  The UCB algorithm selects arms based on upper confidence bounds of the estimated rewards, without considering any context. It is suitable when no contextual information is available.
* **Model**: Estimates the average reward for each arm.
* **Exploration**: Adds a confidence term to the average reward to explore less-tried arms.
* **Exploitation**: Chooses the arm with the highest upper confidence bound.
##### Thompson Sampling
* **Description**: Thompson Sampling is a Bayesian algorithm that selects arms based on samples drawn from the posterior distributions of the arm's reward probabilities.
* **Model**: Assumes Bernoulli-distributed rewards for each arm.
* **Exploration**: Sample from the posterior distributions.
* **Exploitation**: Sample from the posterior distributions.
#### Context-Aware
##### LinUCB
* **Description**: The LinUCB algorithm is a contextual bandit algorithm that uses linear regression to predict the expected reward for each arm given the current context. It balances exploration and exploitation by adding an upper confidence bound to the estimated rewards.
* **Model**: Assumes that the reward is a linear function of the context features.
* **Exploration**: Incorporates uncertainty in the estimation by adding a confidence interval (scaled by alpha).
* **Exploitation**: Chooses the arm with the highest upper confidence bound.
##### KernelUCB
* **Description**: KernelUCB uses kernel methods to capture non-linear relationships between contexts and rewards. It extends the UCB algorithm to a kernelized context space.
* **Model**: Uses a kernel function (e.g., RBF kernel) to compute similarity between contexts.
* **Exploration**: Adds an exploration term based on the uncertainty in the kernel space.
* **Exploitation**: Predicts the expected reward using kernel regression.
### Our Approach

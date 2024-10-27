# Amazon Dataset Rating Prediction
> Final Project for Business Intelligence
## Dataset
This dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review. Note: this dataset contains potential duplicates, due to products whose reviews Amazon merges. A file has been added below (possible_dupes.txt.gz) to help identify products that are potentially duplicates of each other.


For more details, refer to [Amazon Review Dataset (2013)](https://snap.stanford.edu/data/web-Amazon-links.html) 

and [J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.](http://i.stanford.edu/~julian/pdfs/recsys13.pdf)
## Matrix Factorization
### Simon's FunkSVD
[Simon Funk's Blog. Netflix Update: Try This at Home.](https://sifter.org/simon/journal/20061211.html)
### BiasSVD
### Probabilistic Matrix Factorization (PMF)
[Mnih, A., & Salakhutdinov, R. (2007). Probabilistic matrix factorization. In Advances in neural information processing systems (pp. 1257-1264).](https://papers.nips.cc/paper_files/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
### Bayesian Personalized Ranking (BPR)
[Rendle et. al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. The Conference on Uncertainty in Artificial Intelligence.](https://arxiv.org/pdf/1205.2618)

## Deep Neural Network

## Multi-Armed Bandit
Below are detailed descriptions of some exsiting MAB algorithm, indicating how they work.
### Context-free
#### Upper Confidence Bound (UCB)
* **Description**:  The UCB algorithm selects arms based on upper confidence bounds of the estimated rewards, without considering any context. It is suitable when no contextual information is available.
* **Model**: Estimates the average reward for each arm.
* **Exploration**: Adds a confidence term to the average reward to explore less-tried arms.
* **Exploitation**: Chooses the arm with the highest upper confidence bound.
#### Thompson Sampling
* **Description**: Thompson Sampling is a Bayesian algorithm that selects arms based on samples drawn from the posterior distributions of the arm's reward probabilities.
* **Model**: Assumes Bernoulli-distributed rewards for each arm.
* **Exploration**: Sample from the posterior distributions.
* **Exploitation**: Sample from the posterior distributions.
### Context-Aware
#### LinUCB
* **Description**: The LinUCB algorithm is a contextual bandit algorithm that uses linear regression to predict the expected reward for each arm given the current context. It balances exploration and exploitation by adding an upper confidence bound to the estimated rewards.
* **Model**: Assumes that the reward is a linear function of the context features.
* **Exploration**: Incorporates uncertainty in the estimation by adding a confidence interval (scaled by alpha).
* **Exploitation**: Chooses the arm with the highest upper confidence bound.
#### KernelUCB
* **Description**: KernelUCB uses kernel methods to capture non-linear relationships between contexts and rewards. It extends the UCB algorithm to a kernelized context space.
* **Model**: Uses a kernel function (e.g., RBF kernel) to compute similarity between contexts.
* **Exploration**: Adds an exploration term based on the uncertainty in the kernel space.
* **Exploitation**: Predicts the expected reward using kernel regression.
### Our Approach
* Conetext-aware
* Base on LinUCB but **utilize implicit user/item context** and predict on **explicit feedback**
* Cluster similar users into groups to handle cold-start

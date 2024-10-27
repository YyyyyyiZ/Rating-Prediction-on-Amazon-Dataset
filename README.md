# Amazon Dataset Rating Prediction
> Final Project for Business Intelligence
* User rating prediction based on Amazon Review Dataset.
* Various model implementations
  * SVD-Matrix Factorization
      * FunkSVD
      * BiasSVD
      * Probabilistic Matrix Factorization (PMF)
      * Bayesian Personalized Ranking (BPR)
  * Deep Neural Networks
    * Wide & Deep
    * DeepFM
  * Multi-Armed Bandit
## Data
### Dataset
This dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review. Note: this dataset contains potential duplicates, due to products whose reviews Amazon merges. A file has been added below (possible_dupes.txt.gz) to help identify products that are potentially duplicates of each other.


For more details, refer to [Amazon Review Dataset (2013)](https://snap.stanford.edu/data/web-Amazon-links.html) 

and [J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.](http://i.stanford.edu/~julian/pdfs/recsys13.pdf)

### Feature Engineering

## Matrix Factorization
### Simon's FunkSVD
[Simon Funk's Blog. Netflix Update: Try This at Home.](https://sifter.org/simon/journal/20061211.html)
### BiasSVD
BiasSVD is largely based on FunkSVD but introduces some bias terms. 

Thus,the prediction function is:

$\hat{r}_{ui} = \mathbf{p}_u^T \cdot \mathbf{q}_i + b_u + b_i + b$

where $p_u*T$ and $q_i$ denotes the true interaction terms, while $b_u, b_i, b$ denotes the bias terms.
And the new objective function is:

$\min_{\mathbf{P}, \mathbf{Q}, \mathbf{b}} \sum_{(u,i): r_{ui} \neq ?} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( \sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2 + \sum_u b_u^2 + \sum_i b_i^2 \right)$

### Probabilistic Matrix Factorization (PMF)
[Mnih, A., & Salakhutdinov, R. (2007). Probabilistic matrix factorization. In Advances in neural information processing systems (pp. 1257-1264).](https://papers.nips.cc/paper_files/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
### Bayesian Personalized Ranking (BPR)
[Rendle et. al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. The Conference on Uncertainty in Artificial Intelligence.](https://arxiv.org/pdf/1205.2618)

## Deep Neural Network


## Multi-Armed Bandit
> MAB algorithm is especially suitable for **cold-start** problems and **online learning**.
> 
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
The algorithm in this study is designed based on the classic LinUCB framework, where user and item features are treated as contextual vectors. It is assumed that the reward for selecting an item at each time step has a functional relationship with these vectors. During each recommendation, the item with the highest upper confidence bound (UCB), calculated based on the estimated expected reward and confidence interval, is selected. Building on this core concept, we introduce the following innovations tailored to our specific research context and dataset:
#### Implicit Context Vectors
LinUCB uses explicit user and item features as contextual vectors. However, the dataset used in this study does not include explicit descriptions of user and item features. Therefore, we utilize a large language model to analyze all review information for each user and item, extracting implicit feature vectors to serve as contextual inputs for the model.

#### User Clustering
The dataset in this study contains a large number of users with only one recorded interaction, making it difficult to update the expected reward distribution for each item based on a user's past behavior. To address this, we cluster users based on their feature vectors and update the expected reward distribution for items by using the behavior records of users within each cluster.

#### Explicit Feedback
Most contextual bandit-based recommendation algorithms, such as LinUCB, focus primarily on implicit feedback (e.g., click or no-click binary variables). However, our study aims to predict users' explicit feedback (ratings). Therefore, we modify the functional relationship between the reward and the contextual vectors to better fit this explicit feedback setting.

#### Reward Calculation with Partial Feedback
Traditional MABs calculate rewards by executing actions based on the algorithm’s decision, observing the interaction result with the user, and updating accordingly. However, the dataset in this study is log data, which only includes rewards for actions that were taken in the past and logged. Since these actions often differ from those chosen by the algorithm being evaluated, relying solely on this logged data does not allow for a direct assessment of the algorithm's performance. This issue can be considered a special case of the "off-policy evaluation problem" in reinforcement learning.

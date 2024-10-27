import numpy as np
import random


class BPR:
    """
    Parameters:
        lr : Learning rate.
        reg : L2 regularization factor.
        n_epochs : Number of SGD iterations.
        n_factors : Number of latent factors.
        min_rating : Minimum value a rating should be clipped to at inference time.
        max_rating : Maximum value a rating should be clipped to at inference time.
    """

    def __init__(self, n_factors=20, lr=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True,
                 min_rating=1, max_rating=5):
        self.n_factors = n_factors
        self.lr = lr
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors
        self.min_rating = min_rating
        self.max_rating = max_rating

    def train(self, data, sampler, num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        self.init(data)

        print('initial loss = {0}'.format(self.loss()))
        for it in range(num_iters):
            print('starting iteration {0}'.format(it))
            for u, i, j in sampler.generate_samples(self.train):
                self.update_factors(u, i, j)
            print('iteration {0}: loss = {1}'.format(it, self.loss()))

    def init(self, data):
        X = self._preprocess_data(data)

        self.train = data
        self.item_bias = np.zeros(self.n_items)
        self.pu = np.random.random_sample((self.n_users, self.n_factors))
        self.qi = np.random.random_sample((self.n_items, self.n_factors))

        self.create_loss_samples()

    def _preprocess_data(self, X, train=True):
        X = X.copy()

        if train:
            user_ids = X['userId'].unique().tolist()
            item_ids = X['productId'].unique().tolist()

            self.n_users = len(user_ids)
            self.n_items = len(item_ids)

            user_idx = range(self.n_users)
            item_idx = range(self.n_items)

            self.user_mapping_ = dict(zip(user_ids, user_idx))
            self.item_mapping_ = dict(zip(item_ids, item_idx))

        X['u_id'] = X['userId'].map(self.user_mapping_)
        X['i_id'] = X['productId'].map(self.item_mapping_)

        X.fillna(-1, inplace=True)

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        return X[['u_id', 'i_id', 'score']].values

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100 * self.n_users ** 0.5)

        print('sampling {0} <user,item i,item j> triples...'.format(num_loss_samples))
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(self.data, num_loss_samples)]

    def update_factors(self, u, i, j, update_u=True, update_i=True):
        """
        apply SGD update
        """
        update_j = self.update_negative_item_factors

        x = self.item_bias[i] - self.item_bias[j] + np.dot(self.pu[u, :],
                                                           self.qi[i, :] - self.qi[j, :])

        z = 1.0 / (1.0 + np.exp(x))

        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.lr * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.lr * d

        if update_u:
            d = (self.qi[i, :] - self.qi[j, :]) * z - self.user_regularization * self.pu[
                                                                                 u, :]
            self.pu[u, :] += self.lr * d
        if update_i:
            d = self.pu[u, :] * z - self.positive_item_regularization * self.qi[i, :]
            self.qi[i, :] += self.lr * d
        if update_j:
            d = -self.pu[u, :] * z - self.negative_item_regularization * self.qi[j, :]
            self.qi[j, :] += self.lr * d

    def loss(self):
        ranking_loss = 0
        for u, i, j in self.loss_samples:
            x = self.predict(u, i) - self.predict(u, j)
            ranking_loss += 1.0 / (1.0 + np.exp(x))

        complexity = 0
        for u, i, j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.pu[u], self.pu[u])
            complexity += self.positive_item_regularization * np.dot(self.qi[i], self.qi[i])
            complexity += self.negative_item_regularization * np.dot(self.qi[j], self.qi[j])
            complexity += self.bias_regularization * self.item_bias[i] ** 2
            complexity += self.bias_regularization * self.item_bias[j] ** 2

        return ranking_loss + 0.5 * complexity

    def predict(self, u, i):
        return self.item_bias[i] + np.dot(self.pu[u], self.qi[i])


# sampling strategies

class Sampler(object):

    def __init__(self, sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self, data, max_samples=None):
        self.data = data
        self.num_users, self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert (num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self, user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0, self.num_users - 1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0, self.num_items - 1)
        return i

    def num_samples(self, n):
        if self.max_samples is None:
            return n
        return min(n, self.max_samples)


class UniformUserUniformItem(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u, i, j


class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(self, data, max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_data[u].nonzero()[1]
            if len(user_items) == 0:
                # reset user data if it's all been sampled
                for ix in self.local_data[u].indices:
                    self.local_data[u, ix] = self.data[u, ix]
                user_items = self.local_data[u].nonzero()[1]
            i = random.choice(user_items)
            # forget this item so we don't sample it again for the same user
            self.local_data[u, i] = 0
            j = self.sample_negative_item(user_items)
            yield u, i, j


class UniformPair(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            idx = random.randint(0, self.data.nnz - 1)
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            yield u, i, j


class UniformPairWithoutReplacement(Sampler):

    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        idxs = range(self.data.nnz)
        random.shuffle(idxs)
        self.users, self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            self.idx += 1
            yield u, i, j


class ExternalSchedule(Sampler):

    def __init__(self, filepath, index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        f = open(self.filepath)
        samples = [map(int, line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u, i, j in samples[:num_samples]:
            yield u - self.index_offset, i - self.index_offset, j - self.index_offset

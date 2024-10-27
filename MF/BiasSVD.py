import numpy as np


class BiasSVD:
    """
    Parameters:
        lr : Learning rate.
        reg : L2 regularization factor.
        n_epochs : Number of SGD iterations.
        n_factors : Number of latent factors.
        min_rating : Minimum value a rating should be clipped to at inference time.
        max_rating : Maximum value a rating should be clipped to at inference time.

    Attributes:
        user_mapping_ : Maps user ids to their indexes.
        item_mapping_ : Maps item ids to their indexes.
        global_mean : Ratings arithmetic mean.
        pu_ : User latent factors matrix.
        qi_ : Item latent factors matrix.
        bu_ : User biases vector.
        bi_ : Item biases vector.
    """

    def __init__(self, lr=.005, reg=.02, n_epochs=20, n_factors=100, min_rating=1, max_rating=5):

        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, X):
        X = self._preprocess_data(X)
        self.global_mean = np.mean(X[:, 2])
        self._run_sgd(X)

        return self

    def _preprocess_data(self, X):
        """
        Maps user and item ids to their indexes.
        """

        X = X.copy()

        user_ids = X['userId'].unique().tolist()
        item_ids = X['productId'].unique().tolist()

        n_users = len(user_ids)
        n_items = len(item_ids)

        user_idx = range(n_users)
        item_idx = range(n_items)

        self.user_mapping_ = dict(zip(user_ids, user_idx))
        self.item_mapping_ = dict(zip(item_ids, item_idx))

        X['u_id'] = X['userId'].map(self.user_mapping_)
        X['i_id'] = X['productId'].map(self.item_mapping_)

        X.fillna(-1, inplace=True)

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        return X[['u_id', 'i_id', 'score']].values

    def _run_sgd(self, X):
        """
        Runs SGD algorithm, learning model weights.
        """

        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))

        bu, bi, pu, qi = self._initialization(n_users, n_items)

        for epoch_ix in range(self.n_epochs):
            np.random.shuffle(X)
            total_err, bu, bi, pu, qi = self._run_epoch(X, bu, bi, pu, qi)
            print(f"Epoch {epoch_ix + 1}/{self.n_epochs}; Train Error: {total_err/X.shape[0]}")

        self.bu_ = bu
        self.bi_ = bi
        self.pu_ = pu
        self.qi_ = qi

    def predict(self, X, clip=True):
        """
        Returns estimated ratings of several given user/item pairs.
        """
        return [self.predict_pair(u_id, i_id, clip) for u_id, i_id in zip(X['userId'], X['productId'])]

    def predict_pair(self, u_id, i_id, clip=True):
        """
        Returns the model rating prediction for a given user/item pair.
        """
        user_known, item_known = False, False
        pred = self.global_mean

        if u_id in self.user_mapping_:
            user_known = True
            u_ix = self.user_mapping_[u_id]
            pred += self.bu_[u_ix]

        if i_id in self.item_mapping_:
            item_known = True
            i_ix = self.item_mapping_[i_id]
            pred += self.bi_[i_ix]

        if user_known and item_known:
            pred += np.dot(self.pu_[u_ix], self.qi_[i_ix])

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

    def _initialization(self, n_users, n_items):

        bu = np.zeros(n_users)
        bi = np.zeros(n_items)
        pu = np.random.normal(0, .1, (n_users, self.n_factors))
        qi = np.random.normal(0, .1, (n_items, self.n_factors))

        return bu, bi, pu, qi

    def _run_epoch(self, X, bu, bi, pu, qi):
        """
        Runs an epoch, updating model weights (pu, qi, bu, bi).
        """

        total_err = 0
        for i in range(X.shape[0]):
            user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

            # Predict current rating
            pred = self.global_mean + bu[user] + bi[item]
            for factor in range(self.n_factors):
                pred += pu[user, factor] * qi[item, factor]

            err = rating - pred
            total_err += err**2

            # Update biases
            bu[user] += self.lr * (err - self.reg * bu[user])
            bi[item] += self.lr * (err - self.reg * bi[item])

            # Update latent factors
            for factor in range(self.n_factors):
                puf = pu[user, factor]
                qif = qi[item, factor]

                pu[user, factor] += self.lr * (err * qif - self.reg * puf)
                qi[item, factor] += self.lr * (err * puf - self.reg * qif)

        return total_err, bu, bi, pu, qi
import numpy as np
from sklearn.metrics import mean_absolute_error


class PMF:  # Probabilistic MF
    """
    Parameters:
        lr : Learning rate.
        reg : L2 regularization factor.
        n_epochs : Number of SGD iterations.
        n_factors : Number of latent factors.
        num_batches: Number of batches in each epoch (for SGD optimization)
        batch_size: Number of training samples used in each batch (for SGD optimization)
        min_rating : Minimum value a rating should be clipped to at inference time.
        max_rating : Maximum value a rating should be clipped to at inference time.
    """

    def __init__(self, n_factors=10, lr=1, reg=0.1, momentum=0.8, n_epochs=20, num_batches=10, batch_size=10000,
                 min_rating=0, max_rating=5):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, X):
        train = self._preprocess_data(X)
        n_users = len(np.unique(train[:, 0]))
        n_items = len(np.unique(train[:, 1]))

        self.global_mean = np.mean(train[:, 2])
        self.qi_ = 0.1 * np.random.randn(n_items, self.n_factors)  # M x D 正态分布矩阵
        self.pu_ = 0.1 * np.random.randn(n_users, self.n_factors)  # N x D 正态分布矩阵
        self.qi_inc = np.zeros((n_items, self.n_factors))  # M x D 0矩阵
        self.pu_inc = np.zeros((n_users, self.n_factors))  # N x D 0矩阵

        epoch = 0
        while epoch < self.n_epochs:
            epoch += 1
            shuffled_order = np.arange(train.shape[0])
            np.random.shuffle(shuffled_order)

            for batch in range(self.num_batches):
                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])

                batch_UserID = np.array(train[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.pu_[batch_UserID, :], self.qi_[batch_ItemID, :]), axis=1)
                rawErr = pred_out - train[shuffled_order[batch_idx], 2] + self.global_mean

                # Compute gradients
                Ix_User = (2 * np.multiply(rawErr[:, np.newaxis], self.qi_[batch_ItemID, :]) +
                           self.reg * self.pu_[batch_UserID, :])
                Ix_Item = (2 * np.multiply(rawErr[:, np.newaxis], self.pu_[batch_UserID, :]) +
                           self.reg * self.qi_[batch_ItemID, :])

                dqi_ = np.zeros((n_items, self.n_factors))
                dpu_ = np.zeros((n_users, self.n_factors))

                # loop to aggregate the gradients of the same element
                for i in range(self.batch_size):
                    dqi_[batch_ItemID[i], :] += Ix_Item[i, :]
                    dpu_[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.qi_inc = self.momentum * self.qi_inc + self.lr * dqi_ / self.batch_size
                self.pu_inc = self.momentum * self.pu_inc + self.lr * dpu_ / self.batch_size
                self.qi_ = self.qi_ - self.qi_inc
                self.pu_ = self.pu_ - self.pu_inc

            err = mean_absolute_error(self.predict(X, True), X['score'])
            print(f'Epoch {epoch} / {self.n_epochs}; Train Error: {err}')

    def _preprocess_data(self, X):
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

    def predict(self, X, clip=True):
        """
        Returns estimated ratings of several given user/item pairs.
        """
        return [self.predict_pair(u_id, i_id, clip) for u_id, i_id in zip(X['userId'], X['productId'])]

    def predict_pair(self, u_id, i_id, clip=True):
        user_known, item_known = False, False

        pred = self.global_mean
        if u_id in self.user_mapping_:
            user_known = True
            u_ix = self.user_mapping_[u_id]

        if i_id in self.item_mapping_:
            item_known = True
            i_ix = self.item_mapping_[i_id]

        if user_known and item_known:
            pred += np.dot(self.pu_[u_ix], self.qi_[i_ix])

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

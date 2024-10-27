from bandits.algorithm.context_aware.LinUCB import LinUCB
from bandits.algorithm.context_aware.utils import *
import numpy as np
import pickle
import time

if __name__ == "__main__":
    with open('../train_1.pkl', 'rb') as f:
        train = pickle.load(f)
    train = train[:1000]

    unique_product_train = get_unique_item(train['product_id'], train['p_vec_pca'])
    # for i in range(5, 20):
    #     clustered_users, cluster_score = cluster(i, get_unique_user(train['user_id'], train['u_vec']))
    #     print(f"k {i}: {cluster_score}")
    k = 5
    print(f"Step1: Cluster {len(train)} users into {k} groups according to user features")
    clustered_users, cluster_score = cluster(k, get_unique_user(train['user_id'], train['u_vec']))

    # 每个cluster单独进行训练，clustered_train[0]...clustered_train[4]
    clustered_train = {}
    for cluster_id, user_id in clustered_users.items():
        clustered_train[cluster_id] = train[train['user_id'].isin(user_id)].drop(columns=['user_id'])
    clustered_user_vec_mean = {}
    for cluster_id, clustered_feature in clustered_train.items():
        clustered_user_vec_mean[cluster_id] = clustered_feature['u_vec_pca'].mean()

    models = {}
    alpha = 0.1
    for cluster_id, clustered_feature in clustered_train.items():
        clustered_feature.reset_index(drop=True, inplace=True)
        print(f"Step 2-{cluster_id+1}: Train Cluster{cluster_id+1}, Number of records: {len(clustered_feature)}")
        start_time = time.time()
        total_reward = 0
        alg = LinUCB(arms=clustered_train[cluster_id]['product_id'].unique(),
                     n_features=len(clustered_user_vec_mean[cluster_id]),
                     alpha=alpha)
        for i, row in clustered_feature.iterrows():
            context = row['p_vec_pca']
            chosen_arm = alg.select_arm(context, row['product_id'])
            reward = calculate_reward(chosen_arm, row, unique_product_train)
            total_reward += reward
            alg.update(chosen_arm, context, reward)
            if (i + 1) % 1000 == 0:
                avg_reward = total_reward / (i + 1)
                print(f"Round {i + 1}: Average Reward = {avg_reward:.4f}")
        models[cluster_id] = alg
        end_time = time.time()
        print(f"Training time: {end_time - start_time}")

    with open('../test_1.pkl', 'rb') as f:
        test = pickle.load(f)
        print(f"Step3:  Begin predict {len(test)} records")
        # test = test[:100]
        unique_product_test = get_unique_item(test['product_id'], test['p_vec_pca'])
        test.reset_index(drop=True, inplace=True)
        pred_score = []
        for i, row in test.iterrows():
            assigned_cluster = find_best_cluster(clustered_user_vec_mean, row['u_vec_pca'])
            model = models[assigned_cluster]
            one_score = model.predict(row['product_id'], row['p_vec_pca'])
            if one_score:
                pred_score.append(normalize_score(one_score))
            else:
                history = train[train['product_id'] == row['product_id']]
                try:
                    history2 = history[history['user_id'].isin(clustered_train[assigned_cluster]['user_id'])]
                except:
                    history2 = None
                if not history2:
                    pred_score.append(normalize_score(train['score'].mean()))
                else:
                    print("Predict score according to history")
                    pred_score.append(normalize_score(history['score'].mean()))

        mse = np.mean((test['score'] - pred_score) ** 2)
        print(f"MSE on test set: {mse:.4f}")

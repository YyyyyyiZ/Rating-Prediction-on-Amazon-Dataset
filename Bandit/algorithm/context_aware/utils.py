import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def get_unique_user(users, features):
    merged_list = [(l1, l2) for l1, l2 in zip(users, features)]
    seen = {}
    result = []
    for item in merged_list:
        first_dim_value = item[0]
        if first_dim_value not in seen:
            seen[first_dim_value] = True
            result.append(item)
    return result


def get_unique_item(users, features):
    merged_list = [(l1, l2) for l1, l2 in zip(users, features)]
    seen = {}
    result = {}
    for item in merged_list:
        first_dim_value = item[0]
        if first_dim_value not in seen:
            seen[first_dim_value] = True
            result[item[0]] = item[1]
    return result


def cluster(k, user_feature_score):
    features = np.array([feature for _, feature in user_feature_score])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
    clusters = kmeans.labels_
    score = silhouette_score(features, clusters)
    clustered_users = {cluster: [] for cluster in np.unique(clusters)}
    for user_id, cluster in zip([user_id for user_id, _ in user_feature_score], clusters):
        clustered_users[cluster].append(user_id)
    return clustered_users, score


def find_best_cluster(clustered_centroids, one_user):
    distances = {}
    for cluster_id, centroid in clustered_centroids.items():
        distance = np.linalg.norm(one_user[1] - centroid)
        distances[cluster_id] = distance
    assigned_cluster = min(distances, key=distances.get)
    return assigned_cluster


def calculate_reward(arm, row, unique_product):
    if arm == row['product_id']:
        reward = row['score']
    else:
        vec1 = row['p_vec_pca'].reshape(-1, 1)
        vec2 = unique_product[arm].reshape(-1, 1)
        reward = row['score'] * cosine_similarity(vec1, vec2)[0][0]
    return reward


def normalize_score(score):
    if score > 4:
        score = 5
    elif score < 1:
        score = 1
    return score

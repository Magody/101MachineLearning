import numpy as np
from math import pow


def cosine(vector1, vector2):
    # From formula: cos A = (vector1 . vector2) / |vector1|*|vector2|
    vector1_magnitude = np.linalg.norm(vector1)
    vector2_magnitude = np.linalg.norm(vector2)
    return np.dot(vector1, vector2)/(vector1_magnitude * vector2_magnitude)


def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)


def dot_product_with_norms_controlled(vector1, vector2, alfa=0.5):
    # (||vector1||^alfa) * (||vector2||^alfa) * cos(angle between vector1 and vector2)
    return pow(np.linalg.norm(vector1), alfa) * pow(np.linalg.norm(vector2), alfa) * cosine(vector1, vector2)


def euclidean_distance(vector1, vector2):
    # || vector1 - vecto2 || = la raiz de la sumatoria de cada elemento i^2
    return np.linalg.norm(vector1 - vector2)

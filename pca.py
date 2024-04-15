import matplotlib.pyplot as plt
import numpy as np


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        mean = X.mean(axis=0)
        X = X - mean
        U, S, V = np.linalg.svd(X, full_matrices=False)

        self.U = U
        self.S = S
        self.V = V

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        V = self.V.T
        X = data

        mean = X.mean(axis=0)
        X = X - mean

        X_new = X.dot(V[:, :K])

        return X_new

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.

        """
        X = data

        mean = X.mean(axis=0)
        X = X - mean

        V = self.V.T
        s1 = (self.S**2).cumsum() / (self.S**2).sum()
        num_reduced_features = np.sum(s1 < retained_variance) + 1

        X_new = np.dot(X, V[:, :num_reduced_features])

        return X_new

    def get_V(self) -> np.ndarray:
        """Getter function for value of V"""

        return self.V
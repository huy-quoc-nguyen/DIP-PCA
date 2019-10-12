import numpy as np

from lazylog import LazyLog


class PCA(LazyLog):
    def __init__(self, dataset: np.ndarray):
        """
        :param dataset: A dataset, which is a matrix with the shape of (N x M), where:
                - N: number of samples
                - M: number of features
        """
        # Get the shape of the input data
        super().__init__()
        assert len(dataset.shape) == 2
        self._n_samples, self._n_features = dataset.shape
        self.logger.info({
            'msg': 'Shape of input data',
            'value': dataset.shape
        })

        # Change mean to zero
        self._mean = np.mean(dataset, axis=0)
        self._dataset = dataset - self._mean

        # Calculate covariance matrix of features
        self._covariance_matrix = (1 / (self._n_samples - 1)) * np.matmul(dataset.transpose(), self._dataset)
        assert self._covariance_matrix.shape == (self._n_features, self._n_features)
        self.logger.debug({
            'msg': 'Covariance matrix',
            'shape': self._covariance_matrix.shape,
            'matrix': self._covariance_matrix.tolist()
        })

        # Find eigenvectors and eigenvalues of the covariance matrix (which are the candidates of principal components)
        # Notes: technically in this context, eigenvectors are column vectors
        self._eigenvalues, self._eigenvectors = np.linalg.eig(self._covariance_matrix)
        self._component_indices = np.flip(np.argsort(self._eigenvalues))
        self._eigenvalues = self._eigenvalues[self._component_indices]
        self._eigenvectors = self._eigenvectors[:, self._component_indices]
        for i in range(self._n_features):
            self.logger.debug({
                'msg': '{}-th component'.format(i),
                'value': self._eigenvalues[i],
                'vector': self._eigenvectors[:, i].tolist()
            })
        self.logger.debug({
            'msg': 'Assert orthogonal characteristics',
            'basis': np.matmul(self._eigenvectors, self._eigenvectors.transpose()).tolist()
        })

    def project(self, vectors: np.ndarray, n_reduced_features: int) -> np.ndarray:
        """
        :param vectors: A dataset, which is a matrix with the shape of (N x M), where:
                - N: number of samples
                - M: number of features
        :param n_reduced_features: An integer, which is number of features to be reduced
        """
        # Assert shape of vectors
        assert len(vectors.shape) == 2
        assert vectors.shape[1] == self._n_features
        assert n_reduced_features < len(self._eigenvalues)
        return np.matmul(vectors - self._mean, self._eigenvectors[:, :n_reduced_features])

    def project_and_restore(self, vectors: np.ndarray, n_reduced_features: int) -> np.ndarray:
        """
        :param vectors: A dataset, which is a matrix with the shape of (N x M), where:
                - N: number of samples
                - M: number of features
        :param n_reduced_features: An integer, which is number of features to be reduced
        """
        projected = self.project(vectors, n_reduced_features)
        return np.matmul(projected, self._eigenvectors[:, :n_reduced_features].transpose()) + self._mean

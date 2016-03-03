__author__ = 'thk22'

from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
import numpy as np


class CosineMeans(KMeans):

	def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
				 tol=1e-4, verbose=0, random_state=None, copy_x=True, n_jobs=1,
				 **_):

		super(CosineMeans, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init,
										  max_iter=max_iter,  tol=tol, verbose=verbose,
										  n_jobs=n_jobs, random_state=random_state,
										  copy_x=copy_x, precompute_distances=False)

	def fit(self, X, y=None):
		random_state = check_random_state(self.random_state)
		X = self._check_fit_data(X)

		# Init CosineMeans
		if (isinstance(self.init, np.ndarray)):
			self.cluster_centers_ = self.init
		elif (self.init == 'random'):
			idx = random_state.random_integers(0, X.shape[0]-1, (self.n_clusters,))
			self.cluster_centers_ = X[idx]
		elif (self.init == 'kmeans++'):
			self.cluster_centers_ = self._kmeanspp()
		else:
			raise ValueError('Unknown param passed to `init`: {}. Allowed values are "random", "kmeans++" or an ndarray')

		# Run CosineMeans


	def fit_predict(self, X, y=None):
		pass

	def _kmeanspp(self):
		pass
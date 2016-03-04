__author__ = 'thk22'

from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
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
			self.cluster_centers_ = X[idx].A if sparse.issparse(X) else X[idx]
		elif (self.init == 'kmeans++'):
			self.cluster_centers_ = self._kmeanspp()
		else:
			raise ValueError('Unknown param passed to `init`: {}. Allowed values are "random", "kmeans++" or an ndarray')

		# Run CosineMeans
		centroids = np.zeros((self.n_clusters, X.shape[1]))#sparse.csr_matrix((self.n_clusters, X.shape[1]))
		for _ in range(self.max_iter):
			clustering, distances = pairwise_distances_argmin_min(X=X, Y=self.cluster_centers_, metric='cosine')
			# http://stackoverflow.com/questions/29629821/sum-over-rows-in-scipy-sparse-csr-matrix

			# Todo: This really needs improvement
			for yi in np.unique(clustering):
				row_idx = np.where(clustering==yi)[0]
				centroids[yi] = np.asarray(X[row_idx].multiply(1/len(row_idx)).sum(axis=0))

			# Convergence check
			if (np.all(np.abs(self.cluster_centers_-centroids) < self.tol)):
				break
			self.cluster_centers_ = centroids
		self.cluster_centers_ = centroids

	def fit_predict(self, X, y=None):
		pass

	def _kmeanspp(self):
		pass


if (__name__ == '__main__'):
	import os

	from apt_toolkit.utils.base import path_utils
	from apt_toolkit.utils import vector_utils
	from discoutils.thesaurus_loader import Vectors

	#vecs = vector_utils.load_vector_cache(os.path.join(path_utils.get_dataset_path(), 'wordsim353', 'good_transformed_vectors', 'wikipedia_lc_1_lemma-False_pos-False_vectors_min_count-10_min_features-50_cds-0.75_k-1_pmi_constant.dill'))
	#vectors = Vectors.from_dict_of_dicts(vecs)

	from sklearn.datasets import make_classification
	X, _ = make_classification()

	cmeans = CosineMeans(n_clusters=3, init='random')

	cmeans.fit(sparse.csr_matrix(X))

	print(cmeans.cluster_centers_)
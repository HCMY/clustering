__author__ = 'thk22'

from scipy import sparse
from scipy.stats import rv_discrete
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_random_state
import numpy as np


class CosineMeans(KMeans):

	def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4,
				 verbose=0, random_state=None, copy_x=True, n_jobs=1, **_):

		super(CosineMeans, self).__init__(n_clusters=n_clusters, init=init, n_init=1,
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
			idx = random_state.randint(X.shape[0], (self.n_clusters,))
			self.cluster_centers_ = X[idx].A if sparse.issparse(X) else X[idx]
		elif (self.init == 'k-means++'):
			self.cluster_centers_ = self._kmeanspp(X=X, random_state=random_state)
		else:
			raise ValueError('Unknown param passed to `init`: {}. Allowed values are "random", "k-means++" or an ndarray')

		# Run CosineMeans
		centroids = np.zeros((self.n_clusters, X.shape[1]))#sparse.csr_matrix((self.n_clusters, X.shape[1]))
		for _ in range(self.max_iter):
			clustering, distances = pairwise_distances_argmin_min(X=X, Y=self.cluster_centers_, metric='cosine')
			# http://stackoverflow.com/questions/29629821/sum-over-rows-in-scipy-sparse-csr-matrix

			# Todo: This really needs improvement
			for yi in np.unique(clustering):
				row_idx = np.where(clustering==yi)[0]

				if (sparse.issparse(X)):
					centroids[yi] = np.asarray(X[row_idx].multiply(1/len(row_idx)).sum(axis=0))
				else:
					centroids[yi] = np.multiply(X[row_idx], 1/len(row_idx)).sum(axis=0
																				)
			# Convergence check
			if (np.all(np.abs(self.cluster_centers_-centroids) < self.tol)):
				break
			self.cluster_centers_ = centroids
		self.cluster_centers_ = centroids
		self.labels_ = clustering

		return self

	def fit_predict(self, X, y=None):
		self.fit(X, y)

		return self.predict(X)

	def predict(self, X):
		clustering, _ = pairwise_distances_argmin_min(X=X, Y=self.cluster_centers_, metric='cosine')

		self.labels_ = clustering

		return self.labels_

	def _kmeanspp(self, X, random_state):
		# Based on: https://en.wikipedia.org/wiki/K-means%2B%2B
		Xp = type(X)(X, shape=X.shape, dtype=X.dtype, copy=True) if sparse.issparse(X) else np.copy(X)

		idx = random_state.randint(X.shape[0], size=(1,))

		centroids = Xp[idx]
		Xp = self.delete_row_csr(Xp, idx) if sparse.issparse(Xp) else np.delete(Xp, idx, axis=0)

		while (centroids.shape[0] < self.n_clusters):
			clustering, distances = pairwise_distances_argmin_min(X=Xp, Y=centroids, metric='cosine')

			# Calculate weighted probability distribution
			d = np.power(distances, 2)
			p = d / d.sum()

			dist = rv_discrete(values=(np.arange(Xp.shape[0]), p), seed=random_state)

			# Choose next centroid
			idx = dist.rvs()
			centroids = sparse.vstack((centroids, Xp[idx])) if sparse.issparse(Xp) else np.concatenate((centroids, Xp[idx].reshape(1, -1)), axis=0)

			# Delete center from `Xp`
			Xp = self.delete_row_csr(Xp, idx) if sparse.issparse(Xp) else np.delete(Xp, idx, axis=0)

		return centroids

	def delete_row_csr(self, mat, i): # Courtesy of http://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
		if (not isinstance(mat, sparse.csr_matrix)):
			raise ValueError("works only for CSR format -- use .tocsr() first")

		n = mat.indptr[i+1] - mat.indptr[i]
		if n > 0:
			mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
			mat.data = mat.data[:-n]
			mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
			mat.indices = mat.indices[:-n]
		mat.indptr[i:-1] = mat.indptr[i+1:]
		mat.indptr[i:] -= n
		mat.indptr = mat.indptr[:-1]
		mat._shape = (mat._shape[0]-1, mat._shape[1])

		return mat


if (__name__ == '__main__'):
	import os

	from apt_toolkit.utils.base import path_utils
	from apt_toolkit.utils import vector_utils
	from discoutils.thesaurus_loader import Vectors
	from matplotlib import pyplot as plt
	from scipy.sparse.linalg import svds

	vecs = vector_utils.load_vector_cache(os.path.join(path_utils.get_dataset_path(), 'wordsim353', 'good_transformed_vectors', 'wikipedia_lc_1_lemma-False_pos-False_vectors_min_count-10_min_features-50_cds-0.75_k-1_pmi_constant.dill'))
	vectors = Vectors.from_dict_of_dicts(vecs)

	vectors.init_sims(n_neighbors=10, nn_metric='cosine')

	neighbours = vectors.get_nearest_neighbours('arafat')

	idx = []
	for n, _ in neighbours:
		idx.append(vectors.name2row[n])

	X = vectors.matrix[np.array(idx)]

	X, _, _ = svds(X, 2)

	#from sklearn.datasets import make_classification
	#X, _ = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1)

	cmeans = CosineMeans(n_clusters=2)

	cmeans.fit(X)

	print(cmeans.cluster_centers_)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	Z = cmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
			   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
			   cmap=plt.cm.Paired,
			   aspect='auto', origin='lower')

	plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
	# Plot the centroids as a white X
	centroids = cmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
				marker='x', s=169, linewidths=3,
				color='w', zorder=10)
	plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
			  'Centroids are marked with white cross')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.savefig(os.path.join(path_utils.get_out_path(), 'cosine_means', 'test.png'))
	plt.close()
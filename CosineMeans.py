__author__ = 'thk22'

from sklearn.cluster import KMeans


class CosineMeans(KMeans):

	def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
				 tol=1e-4, verbose=0, random_state=None, copy_x=True, n_jobs=1,
				 **_):

		super(CosineMeans, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init,
										  max_iter=max_iter,  tol=tol, verbose=verbose,
										  n_jobs=n_jobs, random_state=random_state,
										  copy_x=copy_x, precompute_distances=False)

	def fit():
		pass

	def fit_predict():
		pass
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
import numpy as np


class MCL(BaseEstimator, ClusterMixin):
	def __init__(self, exp, gamma, max_iter=300, tol=1e-4, random_state=2906,
				 add_loops=True, delta='I', copy=True):
		self.exp = exp
		self.gamma = gamma
		self.max_iter = max_iter
		self.tol = tol
		if (isinstance(random_state, np.random.RandomState)):
			self.random_state = random_state
		else: # Expect a seed value, fail loudly otherwise
			self.random_state = np.random.RandomState(seed=random_state)
		self.add_loops = add_loops
		self.delta = delta # Assume str, np.array or sparse matrix, fail loudly at some point otherwise
		self.copy = copy

	def _is_diag(self, M): # There must be a better way...
		s = np.sum(np.abs(np.eye(M.shape[0], dtype=np.int8) - (M != 0).astype(np.int8)))

		return s == 0

	def _check_X_delta(self, X):
		if (not X.shape[0] == X.shape[1]):
			raise ValueError('Matrix `X` must be square!')

		if (isinstance(self.delta, str) and self.delta == 'I'):
			self.delta = np.eye(X.shape[0], dtype=np.int8)

		if (self.delta.shape != X.shape):
			raise ValueError('Matrix `delta` must be diagonal and of same shape as X!')

		if (np.any(self.delta) < 0):
			raise ValueError('Matrix `delta` must be Non-negative!')

		if (not self._is_diag(self.delta)):
			raise ValueError('Matrix `delta` must be diagonal!')

	def get_clusters(self, A):
		clusters = []
		xxx = np.diag(A) > 0
		for i, r in enumerate((A > 0).tolist()):
			print('I={}'.format(i))
			print('R={}'.format(r))
			print('R[I]={}'.format(r[i]))
			print('R[I]==XXX[I]: {}'.format(r[i]==xxx[i]))
			if r[i]:
				clusters.append(A[i, :] > 0)
			#print(clusters)
			print('=======================================')
		print(len(clusters))
		clust_map = {}
		for cn, c in enumerate(clusters):
			for x in [i for i, x in enumerate(c) if x]:
				clust_map[cn] = clust_map.get(cn, []) + [x]
		print(clust_map)
		return clust_map

	def fit(self, X):
		X_prime = X if not self.copy else X.copy()

		if (self.add_loops): # Add loops
			self._check_X_delta(X_prime)
			X_prime += self.delta

		# Create Markov Graph
		M = X_prime / X_prime.sum(axis=0)
		np.savetxt('/Users/thomas/Desktop/X.txt', X, fmt='%.6f')

		# Cluster
		for i in range(self.max_iter):
			T = np.linalg.matrix_power(M, self.exp) # expansion

			G = np.power(T, self.gamma)
			M_1 = G / G.sum(axis=0) # contraction

			if (np.all(np.abs(M - M_1) <= self.tol)): # tol check
				break

			M = M_1

		# TODO: Interpret as clusters!!! The production of overlapping clusters is a bit annoying for some interpretations
		print(M)
		np.savetxt('/Users/thomas/Desktop/M.txt', M, fmt='%.8f')
		print('Done: {}!'.format(i))
		self.get_clusters(M)
		return M

if (__name__ == '__main__'):
	import networkx as nx
	from matplotlib import pyplot as plt

	# Create random adjacency matrix
	rnd = np.random.RandomState(seed=1105)
	X = rnd.binomial(n=1, p=0.1, size=(100, 100))

	G = nx.from_numpy_matrix(np.asmatrix(X))
	nx.draw(G)
	plt.savefig('/Users/thomas/Desktop/X.png')

	mcl = MCL(exp=2, gamma=2, max_iter=100)
	M = mcl.fit(X)

	G = nx.from_numpy_matrix(np.asmatrix(M))
	nx.draw(G)

	plt.savefig('/Users/thomas/Desktop/M.png')



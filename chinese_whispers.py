import collections
import numpy as np


class ChineseWhispers(object):
	def __init__(self, max_iter, random_state=2906, group_attr='__GROUP__'):

		if (isinstance(random_state, np.random.RandomState)):
			self.random_state = random_state
		else: # Expect a seed value, fail loudly otherwise
			self.random_state = np.random.RandomState(seed=random_state)

		self.max_iter = max_iter
		self.group_attr = group_attr
		self.labels_ = None

	def fit(self, G, y=None):
		N = np.array(G.nodes())

		# Initialise
		for idx, n in enumerate(G.nodes_iter()):
			G.node[n][self.group_attr] = idx

		# Cluster
		for i in range(self.max_iter):
			self.random_state.shuffle(N)

			for n in N:
				G.node[n][self.group_attr] = self._most_common_class(G=G, neighbours=G.neighbors(n))

		# Label
		self.labels_ = self._labelling(G=G, N=N)

		return self

	def predict(self, G):
		return self._labelling(G=G, N=np.array(G.nodes()))

	def fit_predict(self, G, y=None):
		self.fit(G=G, y=y)

		return self.predict(G)

	def _labelling(self, G, N):
		labels = np.zeros((len(N),), dtype=np.uint32)
		for idx, n in enumerate(G.nodes_iter()):
			labels[idx] = G.node[n][self.group_attr]

		return labels

	def _most_common_class(self, G, neighbours):
		c = collections.defaultdict(int)

		for n in neighbours:
			c[G.node[n][self.group_attr]] += 1

		return max(c, key=c.get)

if (__name__ == '__main__'):
	import networkx as nx

	# Create random adjacency matrix
	rnd = np.random.RandomState(seed=1105)
	X = rnd.binomial(n=1, p=0.2, size=(10, 10))

	# Create graph from adjacency matrix
	G = nx.from_numpy_matrix(np.asmatrix(X))

	# Cluster graph
	cw = ChineseWhispers(max_iter=5)
	y = cw.fit_predict(G)

	print(y)

### Collection of NLP clustering algorithms

Currently implemented are:

	* KMeans with cosine distances
	* Chinese Whispers
	
##### Installation

Its not (yet) uploaded to PyPI, so installation has to be done from source:

	git clone https://github.com/tttthomasssss/clustering.pip
	cd clustering/
	pip install -e .

##### Cosine-Means a.k.a. Spherical KMeans or just KMeans with cosine distance

Because `sklearns` `KMeans` implementation doesn't allow different distance metrics.

Kind of an intermediary between standard `KMeans` and `SpectralClustering` with `affinity='cosine'` (`SpectralClustering` first performs a low-dimensional embedding of the affinity matrix and clusters it, whereas `CosineMeans` works directly on the data matrix).


###### Usage:
	from cosine_means import CosineMeans
	from sklearn.datasets import make_classification
	X, _ = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1)

	cmeans = CosineMeans(n_clusters=2)
	cmeans.fit(X)

	print(cmeans.cluster_centers_)

##### Chinese Whispers

**Note:** requires [networkX](http://networkx.github.io)

see [Biemann (2006) - Chinese Whispers - an Efficient Graph Clustering Algorithm and its Application to Natural Language Processing Problems](http://aclweb.org/anthology/W/W06/W06-3812.pdf)

###### Usage:
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





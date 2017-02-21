#### Cosine-Means a.k.a. Spherical KMeans or just KMeans with cosine distance

Because `sklearns` `KMeans` implementation doesn't allow different distance metrics.

Kind of an intermediary between standard `KMeans` and `SpectralClustering` with `affinity='cosine'` (`SpectralClustering` first performs a low-dimensional embedding of the affinity matrix and clusters it, whereas `CosineMeans` works directly on the data matrix).

###### Installation

Its not (yet) uploaded to PyPI, so installation has to be done from source:

	git clone https://github.com/tttthomasssss/clustering.pip
	cd clustering/
	pip install -e .
	
###### Usage:
	from cosine_means import CosineMeans
	from sklearn.datasets import make_classification
	X, _ = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1)

	cmeans = CosineMeans(n_clusters=2)
	cmeans.fit(X)

	print(cmeans.cluster_centers_)

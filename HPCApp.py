# Parallel K-Means Clustering using CPU

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

X = np.random.rand(10000, 2)

start = time.time()

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

end = time.time()

print("Clustering Completed")
print("Execution Time:", end - start, "seconds")

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, s=2)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            color='red')

plt.title("K-Means Clustering")
plt.show()
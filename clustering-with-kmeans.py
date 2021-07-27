# Example - California Housing
# As spatial features, California Housing's 'Latitude' and 'Longitude' make natural candidates for k-means clustering. 
# In this example we'll cluster these with 'MedInc' (median income) to create economic segments in different 
# regions of California.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("../input/fe-course-data/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()
"""
	MedInc	Latitude	Longitude
0	8.3252	37.88	-122.23
1	8.3014	37.86	-122.22
2	7.2574	37.85	-122.24
3	5.6431	37.85	-122.25
4	3.8462	37.85	-122.25
"""
# Since k-means clustering is sensitive to scale, it can be a good idea rescale or normalize data with extreme values. 
# Our features are already roughly on the same scale, so we'll leave them as-is.
# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()
"""
	MedInc	Latitude	Longitude	Cluster
0	8.3252	37.88	-122.23	1
1	8.3014	37.86	-122.22	1
2	7.2574	37.85	-122.24	1
3	5.6431	37.85	-122.25	1
4	3.8462	37.85	-122.25	4
"""
# Now let's look at a couple plots to see how effective this was. 
# First, a scatter plot that shows the geographic distribution of the clusters. 
# It seems like the algorithm has created separate segments for higher-income areas on the coasts.
sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);
# Chart showing scatter plot with categories ) through 5) colored
# The target in this dataset is MedHouseVal (median house value). 
# These box-plots show the distribution of the target within each cluster. 
# If the clustering is informative, these distributions should, for the most part, separate across MedHouseVal, 
# which is indeed what we see.
X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);
# Box plot showing distribution of target within each cluster

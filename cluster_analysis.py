

# importing all useful libraries
import sklearn
import pandas as pd
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.vq import kmeans, vq
np.random.seed(1)



def Optimal_clusters(water_data, guess, threshold=0.01):
    optimal_k = 0
    euclidean_distance = []
    for j in range(1, guess ):
        center_points, distance = sc.cluster.vq.kmeans(water_data, j, thresh=0.0001)
        a = water_data.shape[1]
        distance = distance / (water_data.shape[1])
        if len(euclidean_distance) != 0:
            #thresholding method
            if (euclidean_distance[-1] - distance) /euclidean_distance[-1]< threshold:
                optimal_k = j - 1
                break
        euclidean_distance.append(distance)
        idx,_ = sc.cluster.vq.vq(water_data, center_points)
    return (optimal_k)


#importing data and fixing
water_treatment_data=pd.read_csv('water-treatment.data', header=None)
output_pca_path='output_pca.txt'
output_kmeans='output.txt'
water_treatment_data.columns=['Attribute_' + str(j) for j in range(water_treatment_data.shape[1])]
water_treatment_data.drop('Attribute_0', inplace=True, axis=1)
water_treatment_data=water_treatment_data.apply(pd.to_numeric, errors='coerce')


#Data normalization for 1 column
water_treatment_data_test=water_treatment_data
df = pd.DataFrame(water_treatment_data_test)
x = df[['Attribute_1']].values.astype(float)
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data_normal = pd.DataFrame(x_scaled)
print(data_normal)



# filling NA values with mean value of that column
water_treatment_data=water_treatment_data.fillna(water_treatment_data.mean(axis=0))

# standardizing the data by min-max scaling
MinMaxScale=sklearn.preprocessing.data.MinMaxScaler()
X=MinMaxScale.fit_transform(water_treatment_data)


def return_cluster_id(data, optimal_k):
    center_points, distance = kmeans(data, optimal_k)
    idx, _ = sc.cluster.vq.vq(data, center_points)
    old_val, new_val = np.unique(idx, return_index=True)
    new_id = idx.copy()
    for j in range(optimal_k):
        new_id[np.where(idx == old_val[j])[0]] = new_val[j]
    vals = dict(zip(np.unique(new_id, return_index=True)[0], np.arange(optimal_k)))
    for j in range(len(new_id)):
        new_id[j] = vals[new_id[j]] + 1

    output = pd.DataFrame([range(1, len(new_id) + 1), new_id]).T
    return (output)



threshold_value=0.01
guess =69#nice
optimum_cluster_num=Optimal_clusters(X, guess, threshold_value)
print('Optimal number of clusters is {}'.format(optimum_cluster_num))
output=return_cluster_id(X, optimum_cluster_num)
# writing to file
output.to_csv(output_kmeans,sep=' ',header=False,index=False)


##PCA

X -= np.mean(X, axis = 0)
cov = np.cov(X, rowvar = False)
eigh_values , eigh_vectors = sc.linalg.eigh(cov)
idx = np.argsort(eigh_values)[::-1]
eigh_vectors = eigh_vectors[:, idx]
eigh_values = eigh_values[idx]

# plotting variance of X explained by eigen vectors
plt.plot(np.cumsum(eigh_values) / np.sum(eigh_values), 'bo-', color='r')
plt.xlabel('Number of attributes')
plt.ylabel('Amount of variables explained(%)')
plt.show()

#####################################################################################################
# projecting X on 18 eigen values since they explain 95% of the variance
#we can see this on the graph
a = np.dot(X, eigh_vectors[:, 0:18])
a=MinMaxScale.fit_transform(a)

threshold_value=0.01
guess=69
optimum_cluster_num=Optimal_clusters(a, guess, threshold_value)
print('Optimal number of clusters is {}'.format(optimum_cluster_num))
output=return_cluster_id(a, optimum_cluster_num)
output.to_csv(output_pca_path,sep=' ',header=False,index=False)




import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

## experimental cancer data
df_all_data = pd.read_csv("~/ghub/df_final_Patientid.csv")
df_all_data = df_all_data.iloc[:,1:]
df_all_data.head(10)
print (df_all_data.shape)
### center data
x_data =df_all_data.iloc[:,0:df_all_data.shape[1]-1]
x_scaled = pd.DataFrame(preprocessing.scale(x_data))
x_scaled.columns = x_data.columns
df_all_data.iloc[:,0:df_all_data.shape[1]-1] = x_scaled
df_all_data.head(10)


## pca analysis

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title(' PCA analysis', fontsize = 20)
finalDf = pd.concat([principalDf, df_all_data['readmit']], axis = 1)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.loc[:,'readmit'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color)
ax.legend(targets)
ax.grid()

## here we see first two components overlap heavily


pca = PCA(n_components=14)
X_train_pca = pca.fit_transform(x_scaled)
plt.bar(range(0, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xticks([], [])
plt.show()


#### LDA analysis

numberoflabels = 2
mean_vecs = []
for label in range(numberoflabels):
    labeldata = df_all_data[df_all_data.readmit == label]
    mean_vecs.append(np.mean(labeldata.iloc[:,0:labeldata.shape[1]-1], axis=0))



## within covariance
d = 166  # number of features
s_w = np.zeros((d, d))
for label in zip(range(numberoflabels)):
    labeldata = df_all_data[df_all_data.readmit == label]
    class_scatter = np.cov(labeldata.iloc[:,0:labeldata.shape[1]-1].T)
    s_w += class_scatter


mean_overall = np.mean(df_all_data.iloc[:,0:df_all_data.shape[1]-1], axis=0)
s_b = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    labeldata = df_all_data[df_all_data.readmit == label]
    n = labeldata.shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))


# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])



tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(0, 166), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(0, 166), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/lda1.png', dpi=300)
plt.show()


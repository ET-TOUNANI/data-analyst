# PCA on the Breast Cancer Wisconsin

In this exercise, I will apply PCA to the Breast Cancer Wisconsin (Diagnostic) dataset, which is a dataset of breast cancer diagnoses. The dataset contains 30 different measurements of the characteristics of cell nuclei in breast mass samples, and our goal will be to use PCA to reduce the dimensionality of the dataset from 30 to 2, so that I can visualize the data in a 2D scatter plot.

## Libraries and functions

I will use the following libraries and functions:

- `pandas` for loading and preprocessing the data

- `sklearn.decomposition.PCA` for applying PCA

- `matplotlib.pyplot` for visualizing the data

- `numpy` for computing the mean and standard deviation of the data

```{python}
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
```

## Load the data

First, I will load the data and take a look at it.

```{python}
# you can either ignore the header row of the CSV file or specify the column names using the name parameter
data = pd.read_csv("input/wdbc.data",header=None)
data.head()
```

## Clean the data

Next, I will clean the data. I will drop the ID column, and I will convert the diagnosis column to a binary variable, where 0 indicates a benign diagnosis and 1 indicates a malignant diagnosis.

```{python}
# how to convert the data of an column
transform =lambda x:0 if x =="M"else 1
data[1]=data[1].apply(transform)
data.head()

# drop the ID
X=data.iloc[:,1:]

```

## Preprocess the data

Next, I will preprocess the data. I will standardize the data, so that each feature has mean 0 and standard deviation 1. I will also split the data into a training set and a test set.

```{python}
def center_reduite(x):
    return (x-x.mean())/np.std(x)

X_cr = X.apply(center_reduite)
X_cr.corr()
```
## Create my own PCA

Now, I will create my own PCA. 


```{python}
def my_pca(X):
    # compute the covariance matrix
    cov = np.cov(X.T)
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # sort the eigenvalues in descending order
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    # return the eigenvalues and eigenvectors
    return eig_vals, eig_vecs

```

## Apply PCA

Now, I will apply PCA to the training set. I will use the `PCA` class from `sklearn.decomposition` to do this. I will set the number of components to 2, so that I can visualize the data in a 2D scatter plot.

```{python}
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cr)
```

## Visualize the data

Finally, I will visualize the data. I will plot the first two principal components of the training set, and color the points according to the diagnosis.

```{python}
plt.scatter(X_pca[:, 0], X_pca[:, 1], c="r")
plt.xlabel('Col1')
plt.ylabel('Col2')
plt.show()
```

## References

- [Breast Cancer Wisconsin (Diagnostic) Data Set](<https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>)

- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

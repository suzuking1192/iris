from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

iris_dataset=load_iris()

X_train,X_test,y_train,y_test =train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)

grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train)
plt.show()

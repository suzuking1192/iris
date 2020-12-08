from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.ensemble import RandomForestClassifier

iris_dataset=load_iris()

X_train,X_test,y_train,y_test =train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)

#grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train)
#plt.show()

model = RandomForestClassifier(n_estimators=5,random_state=0)
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

# find the best number of trees

i= 1
while i <= 10:
    testmodel=RandomForestClassifier(n_estimators=i,random_state=0)
    testmodel.fit(X_train,y_train)
    print("Accuracy : {} trees ={}".format(i,testmodel.score(X_test,y_test)))

    i += 1

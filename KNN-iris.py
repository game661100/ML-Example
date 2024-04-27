from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


data = datasets.load_iris()


# print(data.DESCR)
# print(data.target) #0 = Setosa 1 = Versicolour 2 = Virginica
x = range(50)
# plt.scatter(x,data.data[:50,3],color='red') #petal width of Setosa
# plt.scatter(x,data.data[50:100,3],color='blue') #petal width of Versicolour 
# plt.scatter(x,data.data[100:,3],color='green') #petal width of Virginica

# plt.scatter(x,data.data[:50,2],color='red') #petal length of Setosa
# plt.scatter(x,data.data[50:100,2],color='blue') #petal length of Versicolour 
# plt.scatter(x,data.data[100:,2],color='green') #petal length of Virginica
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(data.data[:,2:4], data.target, test_size=0.2, random_state=42,stratify=data.target)
# print('Train Shape X: {} Y : {}'.format(X_train.shape,y_train.shape))
# print('Test Shape X: {} Y : {}'.format(X_test.shape,y_test.shape))

scaler = StandardScaler()
X = scaler.fit_transform(data.data[:,2:4])
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)

answer = knn.predict(X_test)
print(classification_report(y_test, answer))

y_pred = knn.predict(X_test)
print('ACC ',accuracy_score(y_test, y_pred))
# parameters = {'n_neighbors': range(1,11)}
# knn_best = GridSearchCV(knn, parameters, cv=5)
# knn_best.fit(X_train,y_train)
# knn_best.best_estimator_
# answer = knn.predict(X_test)
# print(classification_report(y_test, answer))
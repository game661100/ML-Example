import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('titanic_data.csv')
# print(df.shape)

# print(df.isnull().sum())

age = df['Age'].values
age = np.reshape(age,(-1,1))

imp = SimpleImputer(missing_values = np.nan , strategy='most_frequent')
imp.fit(age)

df['Age'] = imp.transform(age)
df[df['Age'].isnull()]

# X = df[['Pclass','Fare']]
# y = df['Survived']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LogisticRegression()
# lr.fit(X_train,y_train)
# y_pred = lr.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# print(df.describe())

# ans = lr.predict_proba([[2,32.204208]])
# print(ans[:])

df = pd.concat([df,pd.get_dummies(df['Sex'], prefix='Sex',
                                 dummy_na=True)],axis=1).drop(['Sex'],axis=1)
print(df.head())

# X = df[['Pclass','Fare','Sex_female','Sex_male']]
# y = df['Survived']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# lr.fit(X_train,y_train)
# y_pred = lr.predict(X_test)

# print(accuracy_score(y_test, y_pred))

# conf = confusion_matrix(y_test, y_pred)
# print('                 Confusion matrix')
# print('                          Survie           Die   ')
# print('Actual Survie      %6d' % conf[0,0] + '            %5d' % conf[0,1] )
# print('Actual Die          %6d' % conf[1,0] + '            %5d' % conf[1,1] )

X = df[['Pclass','Fare','Sex_female','Sex_male','Age']]
y = df['Survived']

scale = StandardScaler()
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# lr.fit(X_train,y_train)
# y_pred = lr.predict(X_test)
# from sklearn.metrics import accuracy_score
# print('ACC ',accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

parameters = {'C': np.arange(1,10,0.5)}
lr_best = GridSearchCV(lr, parameters, cv=5)
lr_best.fit(X_train,y_train)
lr_best.best_estimator_

y_pred = lr_best.predict(X_test)
from sklearn.metrics import accuracy_score
print('ACC ',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

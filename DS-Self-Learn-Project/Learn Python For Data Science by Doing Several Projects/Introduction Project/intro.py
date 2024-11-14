from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# defining the classifiers
dt = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier()
nn = MLPClassifier()
lr = LogisticRegression()

# training the models
dt = dt.fit(X, Y)
knn = knn.fit(X, Y)
nn = nn.fit(X, Y)
lr = lr.fit(X, Y)

# evaluating the models
dt_prediction = dt.predict(X)
knn_prediction = knn.predict(X)
nn_prediction = nn.predict(X)
lr_prediction = lr.predict(X)

dt_acc = accuracy_score(Y, dt_prediction) * 100
knn_acc = accuracy_score(Y, knn_prediction) * 100
nn_acc = accuracy_score(Y, nn_prediction) * 100
lr_acc = accuracy_score(Y, lr_prediction) * 100

result = {'df': dt_acc, 'knn' : knn_acc, 'nn' : nn_acc, 'lr' : lr_acc}

for keys, items in result.items():
     print(keys, ':', items)
     
# the best model
best_model = max(result, key=result.get)
print('The best model is:', best_model)
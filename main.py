import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.random as rd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sklearn
import seaborn as sns
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import lime

import lime.lime_tabular
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

col_names = ['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety', 'cls']
data = pd.read_csv("car.data.txt", names=col_names)
data = data.drop(columns='cls')
kl = pd.read_csv("car.data.txt", names=col_names)
kl = kl.drop(columns=['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety'])
# print(data.describe())
# print(data.isna().sum())
# print("buying column: ")
# print(data['buying'])
# kl.value_counts().plot(kind='pie', title='rozkład wartości w poszczególnych klasach', autopct='%1.2f%%')
# plt.ylabel("klasy")
# plt.show()

# Konwertowanie na wartości całkowitoliczbowe
encoder = preprocessing.LabelEncoder()
for i in data.columns:
    data[i] = encoder.fit_transform(data[i])
for i in kl.columns:
    kl[i] = encoder.fit_transform(kl[i])

# Standaryzacja danych
scaler = StandardScaler()
po = scaler.fit_transform(data)
data = pd.DataFrame(data=po, columns=col_names[:-1])

# # TODO WIZUALIZACJA
# label2d, label3d = ('persons', 'buying'), ('persons', 'maint', 'door')
# data2d, data3d = [], []
# for i in range(4):
#     # 0 - acc
#     # 1 - goof
#     # 2 - unacc
#     # 3 - v - good
#     buf = data[data['cls'] == i]
#     data2d.append([buf[i] for i in label2d])
#     data3d.append([buf[i] for i in label3d ])
#
#     # ('persons', 'buying')])
#     # data3d.append([buf[i] for i in ('persons', 'maint', 'door')])
# fig2d, ax2d = plt.subplots(1, 1)
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(projection='3d')
#
# cl = ['acc', 'good', 'unacc', 'v-good']
# for i in range(4):
#     ax2d.scatter(data2d[i][0], data2d[i][1], label = f'{cl[i]}')
#     ax3d.scatter(data3d[i][0], data3d[i][1],data3d[i][2], label = f'{cl[i]}')
#
# ax2d.set_xlabel(label2d[0])
# ax2d.set_ylabel(label2d[1])
# ax2d.set_title(f"Zależność między cechą {label2d[0]} oraz {label2d[1]}")
# ax3d.set_xlabel(label3d[0])
# ax3d.set_ylabel(label3d[1])
# ax3d.set_zlabel(label3d[2])
# ax3d.legend(loc = 'lower left')
# ax2d.legend(loc = 'upper right')
# ax3d.set_title(f"Zależność między cechą {label3d[0]}, {label3d[1]} oraz {label3d[2]}")
# plt.show()


# plt.figure()
# sns.heatmap(data.corr(), annot=True)
# plt.title("Correlation between features")
# plt.show()

# Metody PCA i LDA
pca = PCA(n_components=3).fit_transform(data)
data = pd.DataFrame(data=pca, columns=[i for i in range(1, 4)])
data = pd.concat([data, kl], axis=1)

# plt.figure()
# sns.heatmap(data.corr(), annot=True)
# plt.title("Correlation between features")
# plt.show()


Xfeatures = data[[1, 2, 3]]
ylabels = data['cls']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xfeatures, ylabels, test_size=0.3,
                                                                            random_state=7)

# y_train.value_counts().plot(kind='pie', labels=['unacc', 'vgoof', 'good', 'acc'],
#                             title='rozkład wartości w poszczególnych klasach po podzieleniu\n na zbiór uczący i testowy',
#                             autopct='%1.2f%%')
# plt.ylabel("klasy")
# plt.show()


# # TODO WIZUALIZACJA
# label2d, label3d = ('persons', 'buying'), ('persons', 'maint', 'door')
# data2d, data3d = [], []
# for i in range(4):
#     # 0 - acc
#     # 1 - goof
#     # 2 - unacc
#     # 3 - v - good
#     buf = data[data['cls'] == i]
#     print(len(buf))
#     data2d.append([buf[i] for i in label2d])
#     data3d.append([buf[i] for i in label3d])
#
#     # ('persons', 'buying')])
#     # data3d.append([buf[i] for i in ('persons', 'maint', 'door')])
# fig2d, ax2d = plt.subplots(1, 1)
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(projection='3d')
#
# cl = ['acc', 'good', 'unacc', 'v-good']
# for i in range(4):
#     ax2d.scatter(data2d[i][0], data2d[i][1], label = f'{cl[i]}')
#     ax3d.scatter(data3d[i][0], data3d[i][1],data3d[i][2], label = f'{cl[i]}')
#
# ax2d.set_xlabel(label2d[0])
# ax2d.set_ylabel(label2d[1])
# ax2d.set_title(f"Zależność między cechą {label2d[0]} oraz {label2d[1]}")
# ax3d.set_xlabel(label3d[0])
# ax3d.set_ylabel(label3d[1])
# ax3d.set_zlabel(label3d[2])
# ax3d.legend(loc = 'lower left')
# ax2d.legend(loc = 'upper right')
# ax3d.set_title(f"Zależność między cechą {label3d[0]}, {label3d[1]} oraz {label3d[2]}")
# plt.show()


def meth():
    result = []
    methods_labels = ['Logistic Regression', 'Neural Network', 'SVC', 'KNeighbors', 'Decision Tree']

    # Logistic Regression
    lg = LogisticRegression()
    lg.fit(X_train, y_train)

    # Neural Network
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=5000)
    nn.fit(X_train, y_train)

    # SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    # Decision Tree Classifier
    dt = DecisionTreeClassifier()
    dt = dt.fit(X_train, y_train)

    # print("Logistic Regression Accuarcy Score: ", accuracy_score(y_test, lg.predict(X_test)))
    # print("Neural Network Accuarcy Score: ", accuracy_score(y_test, nn.predict(X_test)))
    # print("SVC Accuarcy Score: ", accuracy_score(y_test, svclassifier.predict(X_test)))
    # print("KNN Accuarcy Score: ", accuracy_score(y_test, knn.predict(X_test)))
    # print("Decision Tree Accuarcy Score: ", accuracy_score(y_test, dt.predict(X_test)))

    result.append(accuracy_score(y_test, lg.predict(X_test)))
    result.append(accuracy_score(y_test, nn.predict(X_test)))
    result.append(accuracy_score(y_test, dt.predict(X_test)))
    result.append(accuracy_score(y_test, svclassifier.predict(X_test)))
    result.append(accuracy_score(y_test, knn.predict(X_test)))

    best_method = result.index(max(result))
    met_plot = plt.bar(methods_labels, result)
    met_plot[best_method].set_color('green')
    plt.title("Accuracy score of each method")
    for i in range(len(result)):
            plt.text(i,result[i],round(result[i], 4))
    plt.show()

meth()
# TODO DODAĆ WYKRES SKUTECZNOSCI WZALEŻ OD ILOSCI SASIADOW
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# model = KNeighborsClassifier(n_neighbors=9)
# model.fit(x_train, y_train)
# acc = model.score(x_test, y_test)
# print(acc)
# predicted = model.predict(x_test)
# names = ["unacc", "acc", "good", "vgood"]
# for x in range(len(predicted)):
#     print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

# n = model.kneighbors([x_test[x]], 9, True)
# print("N: ", n)

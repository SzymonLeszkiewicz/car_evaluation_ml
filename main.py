import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.random as rd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import sklearn
import seaborn as sns
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import lime
import lime.lime_tabular


col_names = ['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety', 'cls']
data = pd.read_csv("car.data.txt", names=col_names)
# print(data.describe())
# print(data.isna().sum())
# print("buying column: ")
# print(data['buying'])
# data['cls'].value_counts().plot(kind = 'pie', title= 'rozkład wartości w poszczególnych klasach', autopct='%1.2f%%')
# plt.show()

encoder = preprocessing.LabelEncoder()
# b = encoder.fit_transform(list(data["buying"]))
# m = encoder.fit_transform(list(data["maint"]))
# d = encoder.fit_transform(list(data["door"]))
# p = encoder.fit_transform(list(data["persons"]))
# l = encoder.fit_transform(list(data["lug_boot"]))
# s = encoder.fit_transform(list(data["safety"]))
# c = encoder.fit_transform(list(data["cls"]))
# print("buying column converted into integar values")
# print(buying)

# X = list(zip(b, m, d, p, l, s))
# y = list(c)
for i in data.columns:
    data[i] = encoder.fit_transform(data[i])

print(data.to_string)
# print(data.dtypes)
# print(data.head())
# plt.figure()
# sns.heatmap(data.corr(), annot=True)
# plt.title("Correlation between features")
# plt.show()
# g = data.describe().to_string()
# print(g)
Xfeatures = data[['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety']]
ylabels = data['cls']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xfeatures, ylabels, test_size=0.3, random_state= 7)

label2d, label3d = ('persons', 'buying'), ('persons', 'maint', 'door')
data2d, data3d = [], []
for i in range(4):
    # 0 - acc
    # 1 - goof
    # 2 - unacc
    # 3 - v - good
    buf = data[data['cls'] == i]
    print(len(buf))
    data2d.append([buf[i] for i in label2d])
    data3d.append([buf[i] for i in label3d])

    # ('persons', 'buying')])
    # data3d.append([buf[i] for i in ('persons', 'maint', 'door')])
fig2d, ax2d = plt.subplots(1, 1)
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')

cl = ['acc', 'good', 'unacc', 'v-good']
for i in range(4):
    ax2d.scatter(data2d[i][0], data2d[i][1], label = f'{cl[i]}')
    ax3d.scatter(data3d[i][0], data3d[i][1],data3d[i][2], label = f'{cl[i]}')

ax2d.set_xlabel(label2d[0])
ax2d.set_ylabel(label2d[1])
ax2d.set_title(f"Zależność między cechą {label2d[0]} oraz {label2d[1]}")
ax3d.set_xlabel(label3d[0])
ax3d.set_ylabel(label3d[1])
ax3d.set_zlabel(label3d[2])
ax3d.legend(loc = 'lower left')
ax2d.legend(loc = 'upper right')
ax3d.set_title(f"Zależność między cechą {label3d[0]}, {label3d[1]} oraz {label3d[2]}")
plt.show()

# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
# model = KNeighborsClassifier(n_neighbors=9)
#
# model.fit(x_train, y_train)
#
# acc = model.score(x_test, y_test)
# print(acc)
#
# predicted = model.predict(x_test)
# names = ["unacc", "acc", "good", "vgood"]
#
# for x in range(len(predicted)):
#     print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
#
# n = model.kneighbors([x_test[x]], 9, True)
# print("N: ", n)

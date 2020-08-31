import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv('heart.csv')
# dataset without target col
x = df.drop('target', 1)
# target col
y = df['target']


# data preprocessing
stdS = StandardScaler()
x = stdS.fit_transform(x)

# PCA and deduction
pca = PCA(n_components=2)
x = pca.fit_transform(x)
# split dataset into train & test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# RandomForestClassifier
model = RandomForestClassifier()
# train data using RandomForest
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
print("RandomForestClassifier:", accuracy)

con = confusion_matrix(y_test, y_predict)
print("\n confusion matrix: \n", con)

print("\n Training classification report \n", classification_report(y_test, y_predict))

# ANN
mlpC = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=500)
mlpC.fit(x_train, y_train)
y_predict1 = mlpC.predict(x_test)

accuracy1 = accuracy_score(y_test, y_predict1)
print("\n ANN:", accuracy1)
con1 = confusion_matrix(y_test, y_predict1)
print("ANN confusion matrix : \n", con1)

print("ANN Training classification report: \n", classification_report(y_test, y_predict1))

# k-Nearest Neighbor(KNN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_predict2 = knn.predict(x_test)
accuracy2 = accuracy_score(y_test, y_predict2)

print("KNN:", accuracy2)
con2 = confusion_matrix(y_test, y_predict2)
print("KNN confusion matrix \n", con2)
print("KNN classification report: \n ", classification_report(y_test, y_predict2))

# SVM
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)
y_predict3 = svm.predict(x_test)
accuracy3 = accuracy_score(y_test, y_predict3)
print("SVM:", accuracy3)
con3 = confusion_matrix(y_test, y_predict3)
print("SVM confusion matrix:\n", con3)
print("\n SVM classification report: \n", classification_report(y_test, y_predict3))

# compare between models
print("best model:", max(accuracy, accuracy1, accuracy2, accuracy3))
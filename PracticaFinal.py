
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as score
import pandas as pd
import numpy as np

# Funcion para calcular la especificidad
def specificity_score(y_test, preds):
    conf_matrix = score.confusion_matrix(y_test, preds)
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    return true_negatives / (true_negatives + false_positives)

print("############################### Animales #############################################")

data = pd.read_csv('.\\zoo.data', header=None)
print(data.head())
print(data.info())

# Dividir los datos en X y Y
X = np.empty((len(data), 16), np.float64)
y = np.empty((len(data)), np.int8)

for i in range(len(data)):
    X[i,0]=data.iat[i, 1] #pelo
    X[i,1]=data.iat[i, 2] #plumas
    X[i,2]=data.iat[i, 3] #huevos
    X[i,3]=data.iat[i, 4] #leche
    X[i,4]=data.iat[i, 5] #airborn
    X[i,5]=data.iat[i, 6] #acuatico
    X[i,6]=data.iat[i, 7] #depredador
    X[i,7]=data.iat[i, 8] #dientado
    X[i,8]=data.iat[i, 9] #vertebradp
    X[i,9]=data.iat[i, 10] #respira
    X[i,10]=data.iat[i, 11] #venenoso
    X[i,11]=data.iat[i, 12] #aletas
    X[i,12]=(float(data.iat[i, 13]))/8.0 #cantidad de piernas normalizado
    #X[i,12]=data.iat[i, 13]
    X[i,13]=data.iat[i, 14] #cola
    X[i,14]=data.iat[i, 15] #domesticado
    X[i,15]=data.iat[i, 16] #tamaño gato
    y[i]=data.iat[i, 17]-1

print("X: ")
print(X)

print("Y: ")
print(y)

# Dividir en sets de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Estandarizar los parametros para knn y svm
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar calificadores
logistic_regression = LogisticRegression(max_iter=5000)
knn = KNeighborsClassifier(n_neighbors=10)
svm = SVC(kernel='linear')
naive_bayes = GaussianNB()

logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Predicciones
logistic_regression_pred = logistic_regression.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)

print("Comprobacion: ")
for i in range(min(len(X_test),10)):
    print(f"Valor real: {y_test[i]}")
    print(f"\tPrediccion Regrecion Logistica: {logistic_regression_pred[i]}")
    print(f"\tPrediccion K Neighbours: {knn_pred[i]}")
    print(f"\tPrediccion SVM: {svm_pred[i]}")
    print(f"\tPrediccion naive bayes: {naive_bayes_pred[i]}")

# Imprimir accuracy
print("\n\nLogistic Regression Accuracy:", score.accuracy_score(y_test, logistic_regression_pred))
print("K-Nearest Neighbors Accuracy:", score.accuracy_score(y_test, knn_pred))
print("Support Vector Machines Accuracy:", score.accuracy_score(y_test, svm_pred))
print("Naive Bayes Accuracy:", score.accuracy_score(y_test, naive_bayes_pred))

print("\nLogistic Regression F1:", score.f1_score(y_test, logistic_regression_pred, average='macro', labels=np.unique(logistic_regression_pred)))
print("K-Nearest Neighbors F1:", score.f1_score(y_test, knn_pred, average='macro', labels=np.unique(knn_pred)))
print("Support Vector Machines F1:", score.f1_score(y_test, svm_pred , average='macro', labels=np.unique(svm_pred)))
print("Naive Bayes F1:", score.f1_score(y_test, naive_bayes_pred, average='macro', labels=np.unique(naive_bayes_pred)))

print("\nLogistic Regression Precision:", score.precision_score(y_test, logistic_regression_pred, average='macro', labels=np.unique(logistic_regression_pred)))
print("K-Nearest Neighbors Precision:", score.precision_score(y_test, knn_pred, average='macro', labels=np.unique(knn_pred)))
print("Support Vector Machines Precision:", score.precision_score(y_test, svm_pred , average='macro', labels=np.unique(svm_pred)))
print("Naive Bayes Precision:", score.precision_score(y_test, naive_bayes_pred, average='macro', labels=np.unique(naive_bayes_pred)))

print("\nLogistic Regression Sensibilidad:", score.recall_score(y_test, logistic_regression_pred, average='macro', labels=np.unique(logistic_regression_pred)))
print("K-Nearest Neighbors Sensibilidad:", score.recall_score(y_test, knn_pred, average='macro', labels=np.unique(knn_pred)))
print("Support Vector Machines Sensibilidad:", score.recall_score(y_test, svm_pred , average='macro', labels=np.unique(svm_pred)))
print("Naive Bayes Sensibilidad:", score.recall_score(y_test, naive_bayes_pred, average='macro', labels=np.unique(naive_bayes_pred)))

print("\nLogistic Regression Especificidad:", score.recall_score(y_test, logistic_regression_pred, average='micro', labels=np.unique(logistic_regression_pred)))
print("K-Nearest Neighbors Especificidad:", score.recall_score(y_test, knn_pred, average='micro', labels=np.unique(knn_pred)))
print("Support Vector Machines Especificidad:", score.recall_score(y_test, svm_pred, average='micro', labels=np.unique(svm_pred)))
print("Naive Bayes Especificidad:", score.recall_score(y_test, naive_bayes_pred, average='micro', labels=np.unique(naive_bayes_pred)))


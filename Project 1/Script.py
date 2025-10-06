# Data Processing


import pandas as pd
data = pd.read_csv("Data/data1.csv")



# Data Visualization & Correlation Analysis

import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
data.hist()
plt.show()



df = pd.DataFrame(data)

arr = df.to_numpy()
# print(arr[:,3])
x = arr[:,0]
y = arr[:,1]
z = arr[:,2]
step = arr[:,3]

plt.figure(2)

plt.subplot(231)
plt.plot(step,x,'.')
plt.xlabel('Step')
plt.ylabel('X-Coordinates')

plt.subplot(232)
plt.plot(step,y,'.')
plt.xlabel('Step')
plt.ylabel('Y-Coordinates')

plt.subplot(233)
plt.plot(step,z,'.')
plt.xlabel('Step')
plt.ylabel('Z-Coordinates')

plt.subplot(234)
plt.plot(x,y,'.')
plt.xlabel('X-Coordinates')
plt.ylabel('Y-Coordinates')

plt.subplot(235)
plt.plot(x,z,'.')
plt.xlabel('X-Coordinates')
plt.ylabel('Z-Coordinates')

plt.subplot(236)
plt.plot(y,z,'.')
plt.xlabel('Y-Coordinates')
plt.ylabel('Z-Coordinates')

plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.4)

plt.show()

import seaborn as sns

corr_matrix = data.corr()
sns.heatmap(np.abs(corr_matrix))

# masked_corr_matrix = np.abs(corr_matrix) < 0.7
# sns.heatmap(masked_corr_matrix)



# Data Prep
from sklearn.model_selection import StratifiedShuffleSplit

my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)

for train_index, test_index in my_splitter.split(data, data["Step"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)

y_train = strat_data_train['Step']
x_train = strat_data_train.drop(columns=['Step'])
y_test = strat_data_test['Step']
x_test = strat_data_test.drop(columns=['Step'])



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score



# Decision Tree
from sklearn import tree

pipeline1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', tree.DecisionTreeClassifier(random_state=42))])

param_grid1 = {
    'model__max_depth': [None, 5, 10, 15],
    'model__min_samples_split': [2, 5, 8],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid1 = GridSearchCV(
    estimator=pipeline1,
    param_grid=param_grid1,
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)

score = cross_val_score(pipeline1, x_train, y_train, cv=cv)
grid1.fit(x_train, y_train)
best1 = grid1.best_estimator_

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Decision Tree')
print("CV scores: ", score)
print("CV score average: ", score.mean())
print("Best params:", grid1.best_params_)
y_pred1 = best1.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred1))
print("Test Precision:", precision_score(y_test, y_pred1, average='macro'))
print("Test F1 Score:", f1_score(y_test, y_pred1, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred1))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
plt.figure(3)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred1),display_labels=grid1.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Decision Tree')
plt.show()



# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))])

param_grid2 = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 5, 10, 15],
    'model__min_samples_split': [2, 5, 8],
    'model__min_samples_leaf': [1, 2, 4],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid2 = GridSearchCV(
    estimator=pipeline2,
    param_grid=param_grid2,
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)

score = cross_val_score(pipeline2, x_train, y_train, cv=cv)
grid2.fit(x_train, y_train)
best2 = grid2.best_estimator_

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Random Forest Classifier')
print("CV scores: ", score)
print("CV score average: ", score.mean())
print("Best params:", grid2.best_params_)
y_pred2 = best2.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred2))
print("Test Precision:", precision_score(y_test, y_pred2, average='macro'))
print("Test F1 Score:", f1_score(y_test, y_pred2, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
plt.figure(4)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred2),display_labels=grid2.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Random Forest')
plt.show()



# SVM Model
from sklearn.svm import SVC

pipeline3 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(random_state=42))])

param_grid3 = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear','rbf','poly'],
    'model__gamma':[0.1, 1, 'scale', 'auto']
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid3 = GridSearchCV(
    estimator=pipeline3, 
    param_grid=param_grid3,
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)

score = cross_val_score(pipeline3, x_train, y_train, cv=cv)
grid3.fit(x_train, y_train)
best3 = grid3.best_estimator_

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('SVM')
print("CV scores: ", score)
print("CV score average: ", score.mean())
print("Best params:", grid3.best_params_)
y_pred3 = best3.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred3))
print("Test Precision:", precision_score(y_test, y_pred3, average='macro'))
print("Test F1 Score:", f1_score(y_test, y_pred3, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred3))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
plt.figure(5)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred3),display_labels=grid3.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('SVM')
plt.show()


# RandomizedSearchCV SVM Model
from sklearn.model_selection import RandomizedSearchCV

param_rdm = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear','rbf','poly'],
    'model__gamma':[0.1, 1, 'scale', 'auto']
}

rdm_svm = RandomizedSearchCV(
    estimator=pipeline3, 
    param_distributions=param_rdm,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)

rdm_svm.fit(x_train, y_train)
best_rdm = rdm_svm.best_estimator_

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Randomized Search SVM')
print("Best params:", rdm_svm.best_params_)
y_pred4 = best_rdm.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred4))
print("Test Precision:", precision_score(y_test, y_pred4, average='macro'))
print("Test F1 Score:", f1_score(y_test, y_pred4, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred4))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
plt.figure(6)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred4),display_labels=rdm_svm.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Randomized Search SVM')
plt.show()



# Model Stacking
# Stacking RFC and SVM
from sklearn.ensemble import StackingClassifier

estimators = [('rfc', RandomForestClassifier())]

SC = StackingClassifier(
    estimators = estimators,
    final_estimator = SVC()
)

SC.fit(x_train, y_train)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Stacking')
y_pred5 = SC.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred5))
print("Test Precision:", precision_score(y_test, y_pred5, average='macro'))
print("Test F1 Score:", f1_score(y_test, y_pred5, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred5))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
plt.figure(7)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred5),display_labels=SC.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Stacking')
plt.show()



# joblib
import joblib

filename = 'trained_rdm_svc_model.sav'

joblib.dump(best_rdm, filename)

load_model = joblib.load('trained_rdm_svc_model.sav')

coords1 = {
    "X": [9.375],
    "Y": [3.0625],
    "Z": [1.51]
}
coords2 = {
    "X": [6.995],
    "Y": [5.125],
    "Z": [0.3875]
}
coords3 = {
    "X": [0],
    "Y": [3.0625],
    "Z": [1.93]
}
coords4 = {
    "X": [9.4],
    "Y": [3],
    "Z": [1.8]
}
coords5 = {
    "X": [9.4],
    "Y": [3],
    "Z": [1.8]
}
random_data1 = pd.DataFrame(coords1)
random_data2 = pd.DataFrame(coords2)
random_data3 = pd.DataFrame(coords3)
random_data4 = pd.DataFrame(coords4)
random_data5 = pd.DataFrame(coords5)

step_pred1 = load_model.predict(random_data1)
step_pred2 = load_model.predict(random_data2)
step_pred3 = load_model.predict(random_data3)
step_pred4 = load_model.predict(random_data4)
step_pred5 = load_model.predict(random_data5)

print("Predicted steps from random coordinates: ")
print(step_pred1)
print(step_pred2)
print(step_pred3)
print(step_pred4)
print(step_pred5)
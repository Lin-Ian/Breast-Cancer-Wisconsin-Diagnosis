# Breast-Cancer-Wisconsin-Diagnosis

A Gradio web application to predict whether breast cancer is benign or malignant.

## Table of Contents
- [Technologies](#technologies)
- [Discoveries](#discoveries)
- [Grid Search with Cross-Validation Best Results](#grid-search-with-cross-validation-best-results)
- [Testing Best Hyperparameters](#testing-best-hyperparameters)

## Technologies
This project is created with:
- Python 3.11
- Gradio 3.39.0
- scikit-learn 1.3.0
- pandas 2.0.3

## Discoveries
By using scikit-learn random forest importance, I was able to reduce the number of features required for the machine learning model from 30 to 5.
(A reduction of 83.3%!)
This helped reduce the complexity of the model and the numbers of fields required by the user to make a prediction.
The significant 83.3% reduction of features only led to a minimal reduction of 1.8% in accuracy, and 4.8% in recall.
Precision remained the same at 100% regardless of whether 5 or 30 features were used to make a prediction.

## Grid Search with Cross-Validation Best Results
| Data    | Model | Mean Fit Time         | STD Fit Time          | Mean Score Time       | STD Score Time        | Hyperparameters                                                           | Split 1 Test Score | Split 2 Test Score | Split 3 Test Score | Split 4 Test Score | Split 5 Test Score | Mean Test Score    | STD Test Score       |
|---------|-------|-----------------------|-----------------------|-----------------------|-----------------------|---------------------------------------------------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------------------|
| Full    | knn   | 0.0                   | 0.0                   | 0.003966522216796875  | 0.004857993863956582  | "{'n_neighbors': 6, 'weights': 'uniform'}"                                | 0.978021978021978  | 0.945054945054945  | 0.978021978021978  | 0.967032967032967  | 0.989010989010989  | 0.9714285714285715 | 0.01490621974313247  |
| Full    | lr    | 0.013358259201049804  | 0.0055709804914079375 | 0.0                   | 0.0                   | "{'C': 10, 'penalty': 'l2'}"                                              | 0.9560439560439561 | 0.945054945054945  | 1.0                | 0.967032967032967  | 0.989010989010989  | 0.9714285714285715 | 0.020381579110979577 |
| Full    | mlp   | 0.20576748847961426   | 0.060526935054504585  | 0.001901721954345703  | 0.003803443908691406  | "{'activation': 'relu', 'hidden_layer_sizes': 100, 'solver': 'lbfgs'}"    | 0.9340659340659341 | 0.945054945054945  | 1.0                | 0.978021978021978  | 0.9560439560439561 | 0.9626373626373625 | 0.02367105409729452  |
| Full    | rf    | 0.010299587249755859  | 0.002628217454044119  | 0.0017894744873046876 | 0.0019049339662006329 | "{'max_depth': 10, 'n_estimators': 5}"                                    | 0.9340659340659341 | 0.9230769230769231 | 0.978021978021978  | 0.945054945054945  | 0.978021978021978  | 0.9516483516483516 | 0.02262775855161975  |
| Full    | svm   | 0.003938722610473633  | 0.003925278890910643  | 0.0                   | 0.0                   | "{'C': 10, 'degree': 2, 'kernel': 'linear'}"                              | 0.967032967032967  | 0.967032967032967  | 1.0                | 0.989010989010989  | 0.978021978021978  | 0.9802197802197803 | 0.012815278889769894 |
| Reduced | knn   | 0.0009965896606445312 | 0.0006302987161901117 | 0.003588104248046875  | 0.0004883440776354491 | "{'n_neighbors': 5, 'weights': 'uniform'}"                                | 0.8901098901098901 | 0.945054945054945  | 0.989010989010989  | 0.945054945054945  | 0.9560439560439561 | 0.945054945054945  | 0.03184917966195484  |
| Reduced | lr    | 0.006371498107910156  | 0.00124188802729829   | 0.0019370079040527343 | 0.0011062638882920217 | "{'C': 0.0001, 'penalty': None}"                                          | 0.9340659340659341 | 0.967032967032967  | 0.978021978021978  | 0.945054945054945  | 0.9560439560439561 | 0.956043956043956  | 0.015540808377726312 |
| Reduced | mlp   | 0.013706541061401368  | 0.004460763410086563  | 0.001766204833984375  | 0.0007025962119599226 | "{'activation': 'identity', 'hidden_layer_sizes': 10, 'solver': 'lbfgs'}" | 0.9340659340659341 | 0.967032967032967  | 0.978021978021978  | 0.945054945054945  | 0.9560439560439561 | 0.956043956043956  | 0.015540808377726312 |
| Reduced | rf    | 0.11216573715209961   | 0.009435538531382046  | 0.004767513275146485  | 0.001135710995475459  | "{'max_depth': 5, 'n_estimators': 100}"                                   | 0.8791208791208791 | 0.9340659340659341 | 0.967032967032967  | 0.9560439560439561 | 0.967032967032967  | 0.9406593406593406 | 0.03304021182059981  |
| Reduced | svm   | 0.0026642322540283204 | 0.0005985527863778658 | 0.0023871898651123048 | 0.0007949355482217832 | "{'C': 0.1, 'degree': 2, 'kernel': 'rbf'}"                                | 0.9230769230769231 | 0.9120879120879121 | 0.978021978021978  | 0.9230769230769231 | 0.967032967032967  | 0.9406593406593406 | 0.02655614499691113  |


## Testing Best Hyperparameters
| Data    | Model | Hyperparameters                                              | Train Time            | Predict Time          | Accuracy | Precision | Recall |
|---------|-------|--------------------------------------------------------------|-----------------------|-----------------------|----------|-----------|--------|
| Full    | RF    | Estimators: 5 / Depth: 10                                    | 0.014785300008952618  | 0.0012280000082682818 | 0.956    | 0.974     | 0.905  |
| Full    | LR    | Penalty: l2 / C: 10                                          | 0.01237879999098368   | 0.001424399990355596  | 0.974    | 1.000     | 0.929  |
| Full    | SVM   | C: 10 / Kernel: linear                                       | 0.0049310000031255186 | 0.0021940000005997717 | 0.974    | 1.000     | 0.929  |
| Full    | KNN   | Neighbors: 6 / Weight: uniform                               | 0.00129489999380894   | 0.18201010001939721   | 0.947    | 0.974     | 0.881  |
| Full    | MLP   | Hidden Layer Size: 100 / Activation: relu / Solver: lbfgs    | 0.12091490000602789   | 0.0011626000050455332 | 0.965    | 0.952     | 0.952  |
| Full    | NB    |                                                              | 0.0027461000136099756 | 0.001266500010387972  | 0.947    | 0.950     | 0.905  |
| Full    | QDA   |                                                              | 0.0031239999807439744 | 0.0009671000007074326 | 0.956    | 0.974     | 0.905  |
| Full    | GP    |                                                              | 0.13100040002609603   | 0.0036312000011093915 | 0.956    | 1.000     | 0.881  |
| Reduced | RF    | Estimators: 100 / Depth: 5                                   | 0.14243740000529215   | 0.00434699998004362   | 0.956    | 1.000     | 0.881  |
| Reduced | LR    | C: 0.0001 / Penalty: None                                    | 0.00963340001180768   | 0.0008977999677881598 | 0.956    | 1.000     | 0.881  |
| Reduced | SVM   | C: 0.1 / Kernel: rbf                                         | 0.004874900041613728  | 0.0026488000294193625 | 0.956    | 1.000     | 0.881  |
| Reduced | KNN   | Neighbors: 5 / Weights: uniform                              | 0.0032614999799989164 | 0.007123899995349348  | 0.947    | 0.974     | 0.881  |
| Reduced | MLP   | Hidden Layer Size: 10 / Activation: identity / Solver: lbfgs | 0.013952700013760477  | 0.0009351000189781189 | 0.956    | 1.000     | 0.881  |
| Reduced | NB    |                                                              | 0.0027779999654740095 | 0.00136160000693053   | 0.956    | 1.000     | 0.881  |
| Reduced | QDA   |                                                              | 0.0023705000057816505 | 0.001132700010202825  | 0.904    | 0.943     | 0.786  |
| Reduced | GP    |                                                              | 0.11300440004561096   | 0.001566199993249029  | 0.956    | 1.000     | 0.881  |

## Acknowledgement
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

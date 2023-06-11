import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {                
                "LogisticRegression": LogisticRegression(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False)
            }
            params = {
                "LogisticRegression": {
                    'solver': ['liblinear', 'lbfgs', 'sag'],
                    'C': [0.1, 1, 10],
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3,5,7],
                    'weights': ['uniform', 'distance'],
                },
                "SVC": {
                    'C': [0.1,1,10],
                    'kernel': ['linear', 'poly', 'rbf'],
                    'gamma': [0.1, 1, 10], 
                },
                "DecisionTreeClassifier": {
                    'criterion': ['gini', 'entropy'],
                   # 'splitter': ['best', 'random'],
                    #'max_features': ['sqrt', 'log2'],
                },
                "AdaBoostClassifier": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "GradientBoostingClassifier": {
                    #'loss': ['deviance', 'exponential'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "RandomForestClassifier": {
                   # 'criterion': ['gini', 'entropy'],
                   # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoostClassifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           models=models, param=params)

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)

            return accuracy, precision, recall, f1

        except Exception as e:
            raise CustomException(e, sys)

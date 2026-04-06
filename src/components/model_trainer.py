import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model Training Part initiated")

            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])


            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False,allow_writing_files=False),
                "AdaBoost Regressor":AdaBoostRegressor(), 
            }
            param_grids = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },
                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300, 500],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "l2_leaf_reg": [1, 3, 5, 7, 9]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2]
                }
            }
            best_models = {}
            model_report = {}

            for model_name, model in models.items():
                if model_name in param_grids:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grids[model_name],
                        n_iter=10,
                        scoring="r2",
                        cv=3,
                        verbose=0,
                        random_state=42,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                else:
                    best_model = model.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                best_models[model_name] = best_model
                model_report[model_name] = score

            best_model_name = max(model_report, key=model_report.get)
            best_model = best_models[best_model_name]
            best_score = model_report[best_model_name]

            print(best_model_name, best_score)
            if best_score<0.6:
                raise CustomException("No best model found")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)


        except Exception as e:
            raise CustomException(e,sys)
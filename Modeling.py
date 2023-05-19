from preprocessor import Preprocessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# import xgboost as xgb


class FeatureSelection:
    def __init__(self, name: str, preprocessor: Preprocessor):
        self.name = name
        self.preprocessor = preprocessor
        self.imp_feat_anx = None

    def random_forest(self):
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.preprocessor.x_features, self.preprocessor.y_features)

        self.imp_feat_anx = pd.DataFrame(self.preprocessor.x_features.columns,
                                         columns=['self.preprocessor.x_features'])
        self.imp_feat_anx['RF_score'] = rf.feature_importances_
        self.imp_feat_anx = self.imp_feat_anx.sort_values(by=['RF_score'],
                                                          ascending=False)
        self.imp_feat_anx = self.imp_feat_anx.head(20)
        # print(self.imp_feat_anx)

    def modeling(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.preprocessor.x_features[
                self.imp_feat_anx['self.preprocessor.x_features'].values],
            self.preprocessor.y_features,
            test_size=0.2,
            random_state=42,
            shuffle=True)
        # print(type(x_train), type(y_train))

        base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Initialize the AdaBoost classifier with the base estimator
        adaboost_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50,
                                            random_state=42)

        # Train the AdaBoost model
        adaboost_model.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = adaboost_model.predict(x_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of Ensemble Model {self.name}:{accuracy}")

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        print(f"F1 score of Ensemble Model {self.name}:{f1}")

        # Calculate precision
        precision = precision_score(y_test, y_pred)
        print(f"Precision of Ensemble Model {self.name}:{precision}")

        # Calculate recall
        recall = recall_score(y_test, y_pred)
        print(f"Recall of Ensemble Model {self.name}:{recall}")

        # Calculate ROC AUC score
        # Convert predicted probabilities to binary predictions
        y_pred = np.round(y_pred)

        # Calculate ROC accuracy
        roc_accuracy = roc_auc_score(y_test, y_pred, average=None)
        print(f"ROC accuracy of Ensemble Model {self.name}:{roc_accuracy}")
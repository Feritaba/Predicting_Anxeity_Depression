from creating_dataframes import Preprocessor, MyDataframe
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Creating an instance of dataframe for anxiety
my_dataframe_anx = MyDataframe()
columns_to_drop_list = ['Anxiety symptoms', 'Panic attack symptoms', 'Depressive symptoms']
my_dataframe_anx.create_target(target_column_name='Anxiety symptoms',
                               columns_to_drop_list=columns_to_drop_list)
preprocessor_anxiety = Preprocessor(my_dataframe_anx)

# Creating an instance of dataframe for depression
my_dataframe_dep = MyDataframe()
my_dataframe_dep.create_target(target_column_name='Depressive symptoms',
                               columns_to_drop_list=columns_to_drop_list)
preprocessor_depression = Preprocessor(my_dataframe_dep)


class FeatureSelection:
    def __init__(self):
        self.imp_feat_anx = None
        self.random_forest()
        self.modeling()

    def random_forest(self):
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(preprocessor_anxiety.x_features, preprocessor_anxiety.y_features)

        self.imp_feat_anx = pd.DataFrame(preprocessor_anxiety.x_features.columns,
                                         columns=['preprocessor_anxiety.x_features'])
        self.imp_feat_anx['RF_score'] = rf.feature_importances_
        self.imp_feat_anx = self.imp_feat_anx.sort_values(by=['RF_score'],
                                                          ascending=False)
        self.imp_feat_anx = self.imp_feat_anx.head(20)
        print(self.imp_feat_anx)

    def modeling(self):
        x_train, x_test, y_train, y_test = train_test_split(
            preprocessor_anxiety.x_features[
                self.imp_feat_anx['preprocessor_anxiety.x_features'].values],
            preprocessor_anxiety.y_features,
            test_size=0.2,
            random_state=42,
            shuffle=True)

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
        print("Accuracy of Ensemble Model:", accuracy)

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        print("F1 score of Ensemble Model:", f1)

        # Calculate precision
        precision = precision_score(y_test, y_pred)
        print("Precision of Ensemble Model:", precision)

        # Calculate recall
        recall = recall_score(y_test, y_pred)
        print("Recall of Ensemble Model:", recall)

        # Calculate ROC AUC score
        # Convert predicted probabilities to binary predictions
        y_pred = np.round(y_pred)

        # Calculate ROC accuracy
        roc_accuracy = (y_test == y_pred).mean()
        print("ROC accuracy of Ensemble Model:", roc_accuracy)


def main():
    feat_select = FeatureSelection()
    feat_select.random_forest()
    feat_select.modeling()


if __name__ == '__main__':
    main()

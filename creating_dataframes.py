import pandas as pd
from imblearn.over_sampling import SMOTE
from impyute.imputation.cs import mice
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class MyDataframe:
    def __init__(self):
        self.df = pd.read_csv('./student.csv')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        self.target = None

    def create_target(self, target_column_name, columns_to_drop_list):
        self.target = self.df[target_column_name]
        self.df.drop(columns_to_drop_list, axis=1, inplace=True)

        # printing the number of rows and columns
        print(f"Total Rows: %s and Columns in {target_column_name}: %s"
              % (self.df.shape[0], self.df.shape[1]))


class Preprocessor:
    def __init__(self, my_dataframe: MyDataframe):

        self.my_dataframe = my_dataframe
        self.missing_encoder = {}
        self.missing_data = None
        self.selected_features = None
        self.features_encoder = {}
        self.features_obj_columns = []
        self.missing_obj_columns = []
        self.features = None
        self.x_features = None
        self.y_features = None
        self.prepare_dataframe()

    def prepare_dataframe(self):

        feat_columns = self.my_dataframe.df.columns[self.my_dataframe.df.notna().all()].tolist()
        self.selected_features = self.my_dataframe.df[feat_columns]
        missing_data_columns = self.my_dataframe.df.columns[
            self.my_dataframe.df.isna().any()].tolist()
        self.missing_data = self.my_dataframe.df[missing_data_columns]
        self.encoding_columns()
        self.imputing_missing_values()
        self.forming_three_feature()
        self.merging_dataframe()
        self.normalizing_oversampling()

    def encoding_columns(self):

        # for target variables
        label_encoder = LabelEncoder()
        self.my_dataframe.target = label_encoder.fit_transform(self.my_dataframe.target)

        # for features
        self.features_obj_columns = list(
            self.selected_features.select_dtypes(include='object').columns)
        self.features_encoder = {}
        for col in self.features_obj_columns:
            le = LabelEncoder()
            self.selected_features.loc[:, col] = le.fit_transform(
                self.selected_features[col].astype(str))
            self.features_encoder[col] = le

        # for missing data
        # because label encoder will encode even nan value, so to preserve it
        # first we filter out nan data, then we train encoder by skipping nan rows,
        # then create copy of that column and transform str data by skipping nan rows
        self.missing_obj_columns = list(self.missing_data.select_dtypes(include='object').columns)
        for col in self.missing_obj_columns:
            le = LabelEncoder()
            nan_mask = self.missing_data[col].isna()
            le.fit_transform(self.missing_data[col][~nan_mask])
            col_copy = self.missing_data[col].copy()
            col_copy.loc[~nan_mask] = le.transform(self.missing_data[col][~nan_mask])
            self.missing_data.loc[:, col] = col_copy.astype('float64')
            self.missing_encoder[col] = le

    def imputing_missing_values(self):
        # mice imputing data
        mice_imputed = mice(self.missing_data.values.astype('float64'))
        self.missing_data = pd.DataFrame(mice_imputed, columns=self.missing_data.columns)

        # rounding predicted values for categorical columns
        categorical_columns = ['Satisfied with living conditions', 'Parental home',
                               'Having only one parent', 'Long commute', 'Mode of transportation',
                               'Private health insurance ', 'Overweight and obesity',
                               'Prehypertension or hypertension', 'Abnormal heart rate',
                               'Vaccination up to date', 'Cigarette smoker (5 levels)',
                               'Cigarette smoker (3 levels)', 'Drinker (3 levels)',
                               'Drinker (2 levels)', 'Marijuana use', 'Other recreational drugs']
        for item in categorical_columns:
            self.missing_data.loc[:, item] = \
                self.missing_data.loc[:, item].round().astype('int')

    def forming_three_feature(self):
        self.missing_data['BMI'] = self.missing_data['Weight (kg)'] / \
                                   (self.missing_data['Height (cm)'] * 0.01 * 2)

        self.missing_data['MAP'] = self.missing_data['Diastolic blood pressure (mmHg)'] + \
                                   (0.412 * (self.missing_data['Systolic blood pressure (mmHg)'] -
                                             self.missing_data['Diastolic blood pressure (mmHg)']) *
                                    self.missing_data['Diastolic blood pressure (mmHg)'])

        self.missing_data['PP'] = self.missing_data['Systolic blood pressure (mmHg)'] - \
                                  self.missing_data['Diastolic blood pressure (mmHg)']

    def merging_dataframe(self):
        # self.features = pd.DataFrame()
        self.features = self.selected_features
        self.features = self.features.join(self.missing_data)
        self.features = self.features.drop(labels=['Weight (kg)',
                                                   'Height (cm)',
                                                   'Systolic blood pressure (mmHg)',
                                                   'Diastolic blood pressure (mmHg)'],
                                           axis=1)

    def normalizing_oversampling(self):
        scale = MinMaxScaler()
        feature_normalizing = self.features[['BMI', 'MAP', 'PP', 'Heart rate (bpm)']]
        normalized = scale.fit_transform(feature_normalizing)
        x = pd.DataFrame(normalized, columns=feature_normalizing.columns)
        self.features = self.features.drop(['BMI', 'MAP', 'PP', 'Heart rate (bpm)'],
                                           axis=1)
        self.features = self.features.join(x)
        smote = SMOTE(random_state=42)
        self.x_features, self.y_features = smote.fit_resample(self.features,
                                                              self.my_dataframe.target)


def main():
    my_dataframe = MyDataframe()
    columns_to_drop_list = ['Anxiety symptoms', 'Panic attack symptoms', 'Depressive symptoms']
    my_dataframe.create_target(target_column_name='Anxiety symptoms',
                               columns_to_drop_list=columns_to_drop_list)
    preprocessor_anxiety = Preprocessor(my_dataframe)

    my_dataframe_dep = MyDataframe()
    my_dataframe_dep.create_target(target_column_name='Depressive symptoms',
                                   columns_to_drop_list=columns_to_drop_list)
    preprocessor_depression = Preprocessor(my_dataframe_dep)

    print(preprocessor_anxiety.x_features.shape, preprocessor_anxiety.y_features.shape)
    print(preprocessor_depression.features[:2])


if __name__ == '__main__':
    main()

from dataframe import MyDataframe
from preprocessor import Preprocessor
from modeling import FeatureSelection


def main():
    print("Starting to predict Anxiety")
    my_dataframe = MyDataframe()
    columns_to_drop_list = ['Anxiety symptoms', 'Panic attack symptoms', 'Depressive symptoms']
    my_dataframe.create_target(target_column_name='Anxiety symptoms',
                               columns_to_drop_list=columns_to_drop_list)
    preprocessor_anxiety = Preprocessor(my_dataframe)

    feat_select_anx = FeatureSelection("Anxiety", preprocessor_anxiety)
    feat_select_anx.random_forest()
    feat_select_anx.modeling()

    print("######################################")
    print("Starting to predict Depression")

    my_dataframe_dep = MyDataframe()
    my_dataframe_dep.create_target(target_column_name='Depressive symptoms',
                                   columns_to_drop_list=columns_to_drop_list)

    preprocessor_depression = Preprocessor(my_dataframe_dep)
    feat_select_dep = FeatureSelection("Depression", preprocessor_depression)
    feat_select_dep.random_forest()
    feat_select_dep.modeling()


if __name__ == '__main__':
    main()

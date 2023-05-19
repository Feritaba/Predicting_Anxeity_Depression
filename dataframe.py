import pandas as pd


# Creating instances of dataframe, Anxiety and Depression
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



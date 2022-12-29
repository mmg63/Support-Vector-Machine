import numpy as np
import os
import pandas as pd

class Datasets:


    def __init__(self, folder_path):

        self.folder_path = folder_path
        self.Name_List = []
        self.Data_List = []

    def read_data(self):

        # read csv dataset file
        for dirpath, dirnames, filenames in os.walk(self.folder_path):

            for f in filenames:
                # save the name of the dataset
                name = self.save_dataset_name(f)
                self.Name_List.append(name)

                path = os.path.join(dirpath, f)
                df = self.read_csv(path)
                self.Data_List.append(df)



    def save_dataset_name(self, file):
        # save the name for each dataset
        name = file.replace(".csv", "")
        return name


    def read_csv(self, file):
        # read dataset csv as a Dataframe
        df = pd.read_csv(file)
        return df


    # ds = ReadDataset('Datasets/moons1.csv')
    # data_ = ds.trainValCreation()
    #
    # print('The end.!')




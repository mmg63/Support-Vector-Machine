from unicodedata import name
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


PLOT_PATH = "Plots/"


class ML_Methods:

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

    def trainValSplit_Kfold(self, dataset, num_repeat=1, num_split=10, random_state=None):
        """
        Create dataset and divide dataset to train and test set with number of folding which user has desired.
        Args:
        ---
            `num_repeat` (`int`, optional): How many times this folding should be repeated. Defaults to 1.
            `num_split` (`int`, optional): Number of folding/ spliting dataset. Defaults to 10.
            `random_state` (`random_state`, optional): The state of Randomization. Defaults to None.

        Return: 4 list of datasets which are splited and folded.

        Example for return:
            out = ds.trainValCreation()
            '''
                out[0][0] --> the first train-set
                ...
                out[0][9] --> the tenth train-set

                out[1][0] --> the first test_set
                ...
                out[1][9] --> the tenth test_set

                out[2][0] --> the first train_targets
                ...
                out[2][9] --> the tenth train_targets

                out[3][0] --> the first test_targets
                ...
                out[3][9] --> the tenth test_targets
            '''
        """
        raw_X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values

        scaler = MinMaxScaler()
        X = scaler.fit_transform(raw_X)

        x_trains = []
        x_tests = []
        y_trains = []
        y_tests = []

        kf = RepeatedKFold(n_splits=num_split, n_repeats=num_repeat, random_state=random_state, shuffle=True)
        for train_index, test_index in kf.split(X):
            # print("Train:", train_index, "Validation:",test_index)
            x_trains.append(X[train_index])
            x_tests.append(X[test_index])
            y_trains.append(Y[train_index])
            y_tests.append(Y[test_index])

        return x_trains, x_tests, y_trains, y_tests

    def preprocess(self, df):
        """
        create x and y from a pandas dataframe
        x, which are 2D point will be scaled using min-max scaler

        :param dataframe:
        :return (Scaled X (minmax), y):
        """

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = MinMaxScaler()
        x = scaler.fit_transform(X)

        return x, y

    def adding_methods(self):
        """
        adding all the methods with their specific names in a list

        :return: a List containing tuple of models (name of the model, model)
        """

        Models = []

        # models
        Models.append(self.SVM_Linear())
        Models.append(self.SVM_rbf())
        Models.append(self.SVM_polynomial())

        return Models

    def Kfold_report(self, Models, x_train, y_train, dataset_name):
        """
        training all the models from the list of models using 10 fold cross validation

        :param x_train:
        :param y_train:
        :return:
        """
        print("**********")
        print("{} Dataset Results: ".format(dataset_name))

        results = []
        method_names = []
        for name, model in Models:
            # train the models
            KFold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            CrossValidation = cross_val_score(model, x_train, y_train, cv=KFold, scoring="accuracy")
            results.append(CrossValidation)
            method_names.append(name)
            print(f"{name} Training Accuracy : {CrossValidation.mean()*100:.2f}%")

        return results, method_names


    def training_models(self, Models, x_train, x_test, y_train, y_test, datasetname):

        for name, model in Models:
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            cm = confusion_matrix(y_test, predicted)
            AS = accuracy_score(y_test, predicted)

            self.confusion_metrics(cm, AS, name, datasetname)

    def SVM_Linear(self):
         """
         create a SVM classifier
         :return (name of the mode, QDA model):
         """
         name = "SVM_Linear"
         SVM_Linear_model = SVC(kernel = 'linear', random_state = 0)
         return(name, SVM_Linear_model)
    
    def SVM_rbf(self):
         """
         create a SVM classifier
         :return (name of the mode, QDA model):
         """
         name = "SVM_rbf"
         SVM_rbf_model = SVC(kernel = 'rbf', random_state = 0)
         return(name, SVM_rbf_model)
    
    def SVM_polynomial(self):
         """
         create a SVM classifier
         :return (name of the mode, QDA model):
         """
         name = "SVM_polynomial"
         SVM_polynomial_model = SVC(kernel = 'poly', degree=2, random_state = 0)
         return(name, SVM_polynomial_model)

    def data_spliting(self, x, y, test_size=0.2, random_state=1):
        """
        Split the data into x_train, x_test, y_train, y_test

        :param x: x (data)
        :param y: y (labels)
        :param test_size: size of test dataset
        :param random_state: 1 or 0
        :return: x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def plotting(self, results, names, dataset_name):

        plt.figure(figsize=(12, 10))
        boxplot = plt.boxplot(results, patch_artist=True, labels=names)


        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'lime', 'grey']
        for box, color in zip(boxplot['boxes'], colors):
            box.set(color=color)

        title = "Classifiers Comparison _ {}".format(dataset_name)
        plt.title(title)

        # saving the plots
        fname = PLOT_PATH + title + ".png"
        plt.savefig(fname, dpi=100)
        plt.close('all')


    def confusion_metrics(self, conf_matrix, accuracy_score, method_name, dataset_name):

        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]

        # calculate the sensitivity
        conf_sensitivity = (TP / (float(TP + FN)+ 0.000001))
        # calculate the specificity
        conf_specificity = (TN / (float(TN + FP) + 0.000001))
        # calculate PPV
        ppv = (TP / (float(TP + FP) + 0.000001))
        # calculate NPV
        npv = (TN / (float(TN + FN) + 0.000001))

        print("**************")
        print("Classifier: {} _ Dataset: {}".format(method_name, dataset_name))
        print("PPV:{:.2f} NPV:{:.2f} Sensitivity:{:.2f} Specificity:{:.2f}".format(ppv, npv, conf_sensitivity, conf_specificity))
        print("Accuracy Score for test_set: {:.2f} ".format(accuracy_score))

    def plot_decision_boundary(self, model, X, Y, model_name, dataset_name):

        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        x_in = np.c_[xx.ravel(), yy.ravel()]

        # Predict the function value for the whole gid
        y_pred = model.predict(x_in)
        y_pred = np.round(y_pred).reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, y_pred, cmap="Pastel1")
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap="Pastel2")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())


        title = "Decision boundry of {} on {}". format(model_name, dataset_name)
        plt.title(title)

        # saving the plots
        fname = PLOT_PATH + title + ".png"
        plt.savefig(fname, dpi=100)
        plt.close('all')




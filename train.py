from sklearn.model_selection import train_test_split, PredefinedSplit
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import recall_score

from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from data import *

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

class ournet(nn.Module):
    def __init__(self, input_dim, output_dim, depth=2, width=1000, hidden=None, eps=0.01, bias=True):
        super(ournet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps
        self.bias = bias

        if hidden is None:
            hidden = [width for _ in range(depth - 1)]

        self.hidden = hidden
        self.hidden_layers = nn.ModuleList()  # Use ModuleList to properly register modules

        # Building the layers
        previous_dim = self.input_dim
        for current_dim in self.hidden:
            layer = nn.Linear(previous_dim, current_dim, bias=self.bias)
            layer.weight = nn.Parameter(torch.randn(layer.weight.shape) * eps, requires_grad=True)
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.zeros_like(layer.bias), requires_grad=True)
            self.hidden_layers.append(layer)
            previous_dim = current_dim

        # Output layer
        final_layer = nn.Linear(self.hidden[-1], self.output_dim, bias=self.bias)
        final_layer.weight = nn.Parameter(torch.randn(final_layer.weight.shape) * eps, requires_grad=True)
        self.hidden_layers.append(final_layer)

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = nn.ReLU()(layer(x))
        return self.hidden_layers[-1](x)



def train_builtin(model,train_features,train_labels,test_features,test_labels):
    model.fit(train_features, train_labels)
    predict = model.predict(test_features)
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)

    recall = recall_score(test_labels, predict)

    print("Recall:", recall)
    print(f"f1_socre: {f1score}, auroc: {auroc}, accuracy: {accuracy}")
    return f1score,model

def find_param(model,param_grid, df):


    # grid_search = GridSearchCV(model, param_grid, cv=3,scoring='f1')  # cv is the number of folds, scoring can be adjusted
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)


    X = df[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values
    y = df['DIQ010']
    # Fit the model to the data
    # grid_search.fit(X, y)

    random_search.fit(X,y)

    # Best parameters found
    print("Best parameters:", random_search.best_params_)

def find_param_for_net(train_df, test_df, weight1, depths, widths, epochs=500, lr=0.01):

        train_dataset = Customdataset(train_df)
        test_dataset = Customdataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
        test_features = test_df[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        test_labels = test_df['DIQ010'].values

        best_f1_score = 0
        best_config = None
        best_model = None
        for depth in depths:
            for width in widths:
                print(f'Training model with depth {depth} and width {width}')
                model = ournet(9, 1, depth,
                               width)  # Assuming input_dim=9, output_dim=1 for binary classification
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight1))
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        labels = labels.reshape((-1, 1))
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                # Evaluation loop
                model.eval()
                predict = torch.sigmoid(
                    model(torch.tensor(test_features).float()).detach())

                predict = (predict >= 0.5).float()

                # 3, 200




                f1score = metrics.f1_score(test_labels, predict)

                print(f'Depth: {depth}, Width: {width}, F1 Score: {f1score}')


                if f1score > best_f1_score:
                    best_f1_score = f1score
                    best_config = (depth, width)
                    best_model = model

        return best_config, best_f1_score, best_model


def train_big_model(train_df,test_df):



    f1=[]
    m=[]

    train_df=train_df.reset_index()
    test_df=test_df.reset_index()
    train_features1=train_df[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values
    train_labels1=train_df['DIQ010'].values
    test_features=test_df[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values
    test_labels = test_df['DIQ010'].values




    # The data is 15% vs 85% need to balance the data
    smote = SMOTE(random_state=42)
    train_features, train_labels = smote.fit_resample(train_features1, train_labels1)

    # check if it is balanced

    # train_labels_series = pd.Series(train_labels)
    # class_percentages = train_labels_series.value_counts(
    #     normalize=True) * 100
    # print(class_percentages)




    # # print(df['DIQ010'].mean(),'are label 1')
    weight0=len(df['DIQ010'])/(2*len(df[df['DIQ010']==0]))
    weight1 = len(df['DIQ010']) / (2 * len(df[df['DIQ010'] == 1]))
    train_weights=[weight0 if label==0 else weight1 for label in train_labels]
    test_weights = [weight0 if label == 0 else weight1 for label in test_labels]

    # print('baseline: guessing 0')
    predict = np.zeros(test_labels.shape)
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)


    print(f1score, auroc, accuracy)

    print('knn')
    knn=KNeighborsClassifier(metric="euclidean", weights="distance", n_neighbors=71)
    #
    # # param_grid = {
    # #     'n_neighbors': range(2, 21),  # Testing values from 1 to 20
    # #     'weights': ['uniform', 'distance'],
    # #     'metric': ['euclidean', 'manhattan', 'minkowski']
    # # }
    # # find_param(knn,param_grid,train_features, train_labels)
    #
    a,b=train_builtin(knn,train_features,train_labels,test_features,test_labels)
    f1.append(a)
    m.append(b)


    print(a, b, accuracy)


    print('random forest')
    # randomforest=RandomForestClassifier()

    randomforest=RandomForestClassifier(n_estimators=100, max_depth=10)

    # # # randomforest = RandomForestClassifier(bootstrap = False,
    # #                      criterion =  'entropy',
    # #                      max_depth = None,
    # #                      max_features = 'sqrt',
    # #                      min_samples_leaf = 2,
    # #                      min_samples_split = 9,
    # #                      n_estimators = 115,
    # #                      random_state = 47)
    #
    # # param_grid = {
    # #     'n_estimators': [100, 200, 300],
    # #     # More trees can be better, but take longer to compute
    # #     'max_depth': [10, 20, 30, None],
    # #     # None means max depth not constrained
    # #     'min_samples_split': [2, 5, 10],
    # #     'min_samples_leaf': [1, 2, 4],
    # #     'max_features': ['auto', 'sqrt', 'log2'],
    # #     'bootstrap': [True, False]
    # # }
    # # find_param(randomforest, param_grid,df)
    a, b = train_builtin(randomforest, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)



    print('adaboost')
    ada = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01, n_estimators=200,estimator=DecisionTreeClassifier(max_depth=2))
    # ada = AdaBoostClassifier(algorithm='SAMME')
    a, b = train_builtin(ada, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)


    # param_grid = {
    #     'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
    #     'n_estimators': [50, 100, 200],  # Number of models to iteratively train
    #     'learning_rate': [0.01, 0.1, 1],  # Controls the contribution of each model
    #     'algorithm': ['SAMME']  # Algorithms to use
    # }
    # find_param(ada, param_grid, df)


    print('logistic regression')
    # regression=LogisticRegression(class_weight='balanced')
    regression = LogisticRegression(class_weight='balanced', C=0.0001, max_iter=1000, penalty="l2", solver="lbfgs")

    a, b = train_builtin(regression, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)


    # param_grid = [
    #     {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced'], 'max_iter': [1000,10000, 20000, 30000]},
    #     {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced'], 'max_iter': [100, 200, 300]}
    # ]
    # find_param(regression, param_grid, df)
    #


    print('svc rbf')
    svc_rbf=SVC(class_weight='balanced', kernel='rbf', max_iter=20000, C = 0.06, gamma= 0.01)
    a, b = train_builtin(svc_rbf, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)

    # #
    # param_grid = {
    #     'C': [0.1, 1, 10],  # Regularization parameter
    #     'gamma': ['scale', 'auto', 0.01, 0.1, 1]  # Kernel coefficient
    # }
    # find_param(svc_rbf, param_grid, df)

    print('SVC Poly')
    svc_poly = SVC(class_weight='balanced', kernel='poly')
    a, b = train_builtin(svc_poly, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)

    # param_grid = {
    #     'C': [0.1, 1, 10],  # Regularization parameter
    #     'degree': [2, 3, 4],  # Degree of the polynomial
    #     'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
    #     'coef0': [0, 0.5, 1]  # Independent term in kernel function
    # }
    # find_param(svc_poly, param_grid, df)


    print('mlp')
    # mlp=MLPClassifier(max_iter=1000)
    mlp = MLPClassifier(max_iter=1000, solver="sgd",
                        learning_rate='constant',
                        hidden_layer_sizes=(50,50), alpha=0.001,
                        activation='tanh')
    a, b = train_builtin(mlp, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)

    # param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Different configurations of layers
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
    #     'learning_rate': ['constant', 'adaptive']
    # }
    # find_param(mlp, param_grid, df)


    train_dataset=Customdataset(train_df)
    test_dataset=Customdataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    print('ournet')

    # depths = [5,7]  # Example depths
    # widths = [5,50, 100, 200,1000]  # Example widths


    # best_config, best_f1, model = find_param_for_net(train_df,test_df,weight1,depths, widths)
    # train_dataset = Customdataset(train_df)
    # test_dataset = Customdataset(test_df)
    # train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    # test_features = test_df[
    #     ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
    #      'WHQ040', 'SLD012', 'OCQ180']].values
    # test_labels = test_df['DIQ010'].values
    #
    #
    model = ournet(9, 1, 3,200)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.reshape((-1, 1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    model.eval()
    predict=torch.sigmoid(model(torch.tensor(test_features).float()).detach())
    predict=(predict>=0.5).float()
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)

    print(f"f1_socre: {f1score}, auroc: {auroc}, accuracy: {accuracy}")
    f1.append(f1score)
    m.append(model)




    # f1_df = pd.DataFrame.from_dict(plot_data, orient='index')
    # f1_df.reset_index(inplace=True)
    # f1_df.rename(columns={'index': 'Race'}, inplace=True)
    # f1_df = f1_df[['Race'] + [col for col in f1_df.columns if col != 'Race']]
    # print(f1_df)
    # f1_df.to_csv(file_name, index=False)

    return m,f1



def train_small_model(train_df_ls, test_df_ls):
    f1_scores = []
    models = []

    for i in range(len(train_df_ls)):

        f1 = []
        m = []
        train_df = train_df_ls[i]
        test_df = test_df_ls[i]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_features1 = train_df[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        train_labels1 = train_df['DIQ010'].values
        test_features = test_df[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        test_labels = test_df['DIQ010'].values

        # The data is 15% vs 85% need to balance the data
        smote = SMOTE(random_state=42)
        train_features, train_labels = smote.fit_resample(train_features1,
                                                          train_labels1)



        # print(df['DIQ010'].mean(),'are label 1')
        weight0 = len(df['DIQ010']) / (2 * len(df[df['DIQ010'] == 0]))
        weight1 = len(df['DIQ010']) / (2 * len(df[df['DIQ010'] == 1]))
        train_weights = [weight0 if label == 0 else weight1 for label in
                         train_labels]
        test_weights = [weight0 if label == 0 else weight1 for label in
                        test_labels]



        print('knn')
        knn = KNeighborsClassifier(metric="euclidean", weights="distance",
                                   n_neighbors=71)

        a, b = train_builtin(knn, train_features, train_labels, test_features,
                             test_labels)
        f1.append(a)
        m.append(b)


        print('random forest')
        # randomforest=RandomForestClassifier()

        randomforest = RandomForestClassifier(n_estimators=100, max_depth=10)


        a, b = train_builtin(randomforest, train_features, train_labels,
                             test_features, test_labels)
        f1.append(a)
        m.append(b)
        #

        print('adaboost')
        ada = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01,
                                 n_estimators=200,
                                 estimator=DecisionTreeClassifier(max_depth=2))
        # ada = AdaBoostClassifier(algorithm='SAMME')
        a, b = train_builtin(ada, train_features, train_labels, test_features,
                             test_labels)
        f1.append(a)
        m.append(b)


        print('logistic regression')
        # regression=LogisticRegression(class_weight='balanced')
        regression = LogisticRegression(class_weight='balanced', C=0.0001,
                                        max_iter=1000, penalty="l2",
                                        solver="lbfgs")

        a, b = train_builtin(regression, train_features, train_labels,
                             test_features, test_labels)
        f1.append(a)
        m.append(b)


        print('svc rbf')
        svc_rbf = SVC(class_weight='balanced', kernel='rbf', max_iter=20000,
                      C=0.06, gamma=0.01)
        a, b = train_builtin(svc_rbf, train_features, train_labels,
                             test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('svc poly')
        svc_poly = SVC(class_weight='balanced', kernel='poly')
        a, b = train_builtin(svc_poly, train_features, train_labels,
                             test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('mlp')
        # mlp=MLPClassifier(max_iter=1000)
        mlp = MLPClassifier(max_iter=1000, solver="sgd",
                            learning_rate='constant',
                            hidden_layer_sizes=(50, 50), alpha=0.001,
                            activation='tanh')
        a, b = train_builtin(mlp, train_features, train_labels, test_features,
                             test_labels)
        f1.append(a)
        m.append(b)

        train_dataset = Customdataset(train_df)
        test_dataset = Customdataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

        print('ournet')


        # best_config, best_f1, model = find_param_for_net(train_df,test_df,weight1,depths, widths)
        train_dataset = Customdataset(train_df)
        test_dataset = Customdataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
        test_features = test_df[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        test_labels = test_df['DIQ010'].values

        model = ournet(9, 1, 3, 200)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight1))
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(100):
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.reshape((-1, 1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        predict = torch.sigmoid(
            model(torch.tensor(test_features).float()).detach())
        predict = (predict >= 0.5).float()
        f1score = metrics.f1_score(test_labels, predict)
        auroc = metrics.roc_auc_score(test_labels, predict)
        accuracy = metrics.accuracy_score(test_labels, predict)

        print(f"f1_socre: {f1score}, auroc: {auroc}, accuracy: {accuracy}")
        f1.append(f1score)
        m.append(model)

        f1_scores.append(f1)
        models.append(m)
    return models, f1_scores


def plot_graphs(df_ls):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    for i,df in enumerate(df_ls):
        for column in ['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']:
            sns.boxplot(data=df,x='RIDRETH3',y=column,showfliers=False)
            plt.title(f'boxplot of {column}_{i}')
            plt.xlabel('Race')
            plt.ylabel(column)
            output_file = os.path.join('plots', f'{column}_{i}_box.png')
            plt.savefig(output_file)
            plt.close()



def plot_bias_graph(df, path, title):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f']

    # Plotting with the new color palette
    ax = df.plot(kind='bar', figsize=(10, 6), color=colors)
    ax.set_title(title)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Model Type')
    plt.xticks(rotation=0)
    plt.legend(title='Model Type')
    plt.tight_layout()

    # Save the plot to a file

    plt.savefig(path)


def plot_bias_box_graph(all_data, category, path):
    colors = ['#1f77b4', '#2ca02c']  # blue and orange

    # Initialize the plot
    fig, ax = plt.subplots()

    # Data preparation for box plots
    data = []
    labels = []

    # Create labels for each pair of 'Big Model' and 'Small Model'
    for name in category:
        labels.append(f"{name} Big Model")
        labels.append(f"{name} Small Model")

    # Collect data for plotting
    for plot_data in all_data:
        big_model_scores = list(plot_data["Big Model"].values())
        small_model_scores = list(plot_data["Small Model"].values())
        data.append(big_model_scores)
        data.append(small_model_scores)

    # Create the box plot
    bp = ax.boxplot(data, patch_artist=True)

    # Assign colors to each boxplot
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 2])  # Alternate between two colors

    # Set x-axis labels to match the number of box plots
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Create custom legends
    big_patch = mpatches.Patch(color=colors[0], label='Big Model')
    small_patch = mpatches.Patch(color=colors[1], label='Small Model')
    plt.legend(handles=[big_patch, small_patch])

    # Set plot titles and labels
    ax.set_title(
        'Comparison of F1 Scores for Big and Small Models Across Datasets')
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Datasets')

    # Adjust the padding between and around subplots.
    plt.tight_layout()

    # Save the figure
    plt.savefig(path)
    plt.close(fig)  # Close the figure to free memory


def ada_boost_bias(df, col_name, category, index):



    big_ada =  AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01, n_estimators=200,estimator=DecisionTreeClassifier(max_depth=2))
    small_ada = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01, n_estimators=200,estimator=DecisionTreeClassifier(max_depth=2))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    x_col = ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']
    y_col = 'DIQ010'

    X_train = train_df[x_col].values
    y_train = train_df[y_col].values

    X_test = test_df[x_col].values
    y_test = test_df[y_col].values

    big_ada.fit(X_train,y_train)
    #这里开始 加
    # y_pred = big_ada.predict(X_test)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # print('confusion matrix')
    # print(conf_matrix)

    # # 分类报告
    # print('classification_report')
    # print(classification_report(y_test, y_pred))

    # # 绘制学习曲线
    # estimator_errors = big_ada.estimator_errors_
    # plt.figure(figsize=(10, 5))
    # plt.plot(estimator_errors, label='Estimator Errors')
    # plt.xlabel('Number of Estimators')
    # plt.ylabel('Error')
    # plt.title('AdaBoost Estimator Errors')
    # plt.legend()
    # plt.show()

    train_small_df = None
    grouped = train_df.groupby(col_name)


    for key, group_df in grouped:
        if key == index:
            train_small_df = group_df
            break

    X_train = train_small_df[x_col].values
    y_train = train_small_df[y_col].values



    small_ada.fit(X_train,y_train)

    big_ada_importances = big_ada.feature_importances_
    small_ada_importances = small_ada.feature_importances_

    print('importances!!!!!!')
    print(big_ada_importances)
    print(small_ada_importances)

    importances_diff = big_ada_importances - small_ada_importances

    feature_names = x_col

    print("Feature importances difference between big and small models:")
    for feature, diff in zip(feature_names, importances_diff):
        print(f"Feature: {feature}, Importance Difference: {diff}")


    indices = range(len(feature_names))

    plt.bar(indices, big_ada_importances, width=0.4, label='Big Model',
            align='center')
    plt.bar([i + 0.4 for i in indices], small_ada_importances, width=0.4,
            label='Small Model', align='center')

    plt.xticks([i + 0.2 for i in indices], feature_names, rotation=45)
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances for Big vs. Small Models')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # DBQ910 ,# of frozen meals/pizza in past 30 days
    # SMD 650#Avg # cigarettes/day during past 30 days
    #Hours worked last week in total all jobs



if __name__ == '__main__':
    df = get_data()
    category = ['Mexican American', 'Hispanic', 'White', 'Black', 'Asian']
    col_name = 'RIDRETH3'
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    ada_boost_bias(train_df, col_name, category, 1)
    ada_boost_bias(train_df, col_name, category, 4)
    #exit()


    combined_models,combined_f1_scores=train_big_model(train_df, test_df)

    # category = ['Poor', 'Medium','Rich' ]
    # col_name = 'INDFMPIR'

    category = ['Mexican American', 'Hispanic', 'White', 'Black', 'Asian']
    col_name = 'RIDRETH3'

    test_dfs = []
    grouped = test_df.groupby(col_name)
    for _, group_df in grouped:
        test_dfs.append(group_df)

    grouped = train_df.groupby(col_name)

    train_dfs = []
    for _, group_df in grouped:
        train_dfs.append(group_df)



    model_names = [
        'KNN',
        'Random Forest',
        'Adaboost',
        'Logistic Regression',
        'SVC RBF',
        'SVC Poly',
        'MLP',
        'Neural Network'
    ]


    # train small model on one race
    print('1 race !!!!!!!!!!!!!!!!!!!!')
    all_data = []
    separated_models, _ = train_small_model(train_dfs, test_dfs)
    for i in range(len(category)):
        test_df_temp = test_dfs[i]
        test_features = test_df_temp[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        test_labels = test_df_temp['DIQ010'].values

        small_models = separated_models[i]
        plot_data = {"Big Model": {}, "Small Model": {}}
        for j in range(7):
            small_model = small_models[j]
            big_model = combined_models[j]

            predict = small_model.predict(test_features)
            small_f1score = metrics.f1_score(test_labels, predict)

            predict = big_model.predict(test_features)
            big_f1score = metrics.f1_score(test_labels, predict)

            plot_data["Big Model"][model_names[j]] =  big_f1score
            plot_data["Small Model"][model_names[j]] = small_f1score

        # neural network
        small_model = small_models[-1]
        small_model.eval()
        predict = torch.sigmoid(
            small_model(torch.tensor(test_features).float()).detach())
        predict = (predict >= 0.5).float()
        small_f1score = metrics.f1_score(test_labels, predict)

        big_model = combined_models[-1]
        big_model.eval()
        predict = torch.sigmoid(
            big_model(torch.tensor(test_features).float()).detach())
        predict = (predict >= 0.5).float()
        big_f1score = metrics.f1_score(test_labels, predict)

        plot_data["Big Model"][model_names[-1]] = big_f1score
        plot_data["Small Model"][model_names[-1]] = small_f1score
        all_data.append(plot_data)
        plot_df = pd.DataFrame.from_dict(plot_data, orient='index')
        plot_bias_graph(plot_df,f"plots/{category[i]}.png", f"{category[i]} f1 score by model")

    plot_bias_box_graph(all_data,category, f"plots/race_boxplot1.png")

    # plot_bias_box_graph(all_data, category, f"plots/wealth_boxplot1.png")




    # train small model on  n-1

    all_data = []
    for i in range(len(category)):


        grouped = df.groupby(col_name)
        separated_df = []

        k = 0

        train_partial_df = train_dfs[:i] + train_dfs[i+1:]
        test_partial_df = test_dfs[:i] + test_dfs[i+1:]

        train_df_temp = pd.concat(train_partial_df, ignore_index=True)
        test_df_temp = pd.concat(test_partial_df, ignore_index=True)

        small_models, _ = train_big_model(train_df_temp, test_df_temp)

        test_df_temp = test_dfs[i]
        test_features = test_df_temp[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        test_labels = test_df_temp['DIQ010'].values
        plot_data = {"Big Model": {}, "Small Model": {}}
        for j in range(7):
            small_model = small_models[j]
            big_model = combined_models[j]

            predict = small_model.predict(test_features)
            small_f1score = metrics.f1_score(test_labels, predict)

            predict = big_model.predict(test_features)
            big_f1score = metrics.f1_score(test_labels, predict)

            plot_data["Big Model"][model_names[j]] = big_f1score
            plot_data["Small Model"][model_names[j]] = small_f1score

        # neural network
        small_model = small_models[-1]
        small_model.eval()
        predict = torch.sigmoid(
            small_model(torch.tensor(test_features).float()).detach())
        predict = (predict >= 0.5).float()
        small_f1score = metrics.f1_score(test_labels, predict)

        big_model = combined_models[-1]
        big_model.eval()
        predict = torch.sigmoid(
            big_model(torch.tensor(test_features).float()).detach())
        predict = (predict >= 0.5).float()
        big_f1score = metrics.f1_score(test_labels, predict)

        plot_data["Big Model"][model_names[-1]] = big_f1score
        plot_data["Small Model"][model_names[-1]] = small_f1score
        plot_df = pd.DataFrame.from_dict(plot_data, orient='index')
        plot_bias_graph(plot_df, f"plots/{category[i]}_2.png",
                        f"{category[i]} f1 score by model (n-1 version)")
        all_data.append(plot_data)

    plot_bias_box_graph(all_data, category, f"plots/race_boxplot2.png")

    # plot_bias_box_graph(all_data, category, f"plots/wealth_boxplot2.png")


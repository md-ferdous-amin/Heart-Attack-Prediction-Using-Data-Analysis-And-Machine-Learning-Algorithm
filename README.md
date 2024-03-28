<b> Heart Attack Prediction </b>

This code explores a dataset on heart disease and uses various machine learning models to predict the presence of heart disease.

<b>Libraries</b>

    numpy https://numpy.org/doc/stable/
    pandas https://pandas.pydata.org/
    missingno https://github.com/Lei-Cai/PYTHON-missingno
    seaborn https://seaborn.pydata.org/
    matplotlib https://matplotlib.org/
    plotly https://plotly.com/python/
    wordcloud https://pypi.org/project/wordcloud/
    scikit-learn https://scikit-learn.org/

<b>Data</b>

The code assumes two CSV files are located in the specified directory:

    heart.csv - Contains data on heart disease patients
    o2Saturation.csv - Contains oxygen saturation data

<b>Code walkthrough</b>

    Import libraries - The necessary libraries for data manipulation, visualization, and machine learning are imported.
    Data loading - The code reads the two CSV files using pandas.read_csv().
    Data Cleaning - The oxygen saturation data is renamed for better readability. The dataframes are then concatenated.
    Exploratory Data Analysis (EDA) -
        Checks for missing values using data.isnull().sum().
        Generates summary statistics using data.describe().
        Provides information on data types and non-null values using data.info().
        Shows the shape of the data using data.shape.
        Lists column names using data.columns.
    Visualization -
        Creates a distribution plot for the "age" feature using seaborn.displot().
        Creates count plots for categorical features using seaborn.countplot().
    Machine Learning Model Training -
        Splits the data into training and testing sets using scikit-learn's train_test_split().
        Defines a list of classifiers including Decision Tree, SVM, Random Forest, Logistic Regression, and K-Nearest Neighbors.
        Defines parameter grids for each classifier to tune hyperparameters using GridSearchCV.
        Performs stratified K-fold cross-validation with 10 folds using StratifiedKFold().
        Uses GridSearchCV to find the best hyperparameters for each model that maximizes accuracy.
        Evaluates the performance of the models on the training and testing sets using accuracy_score().
    Results -
        Prints the training and testing accuracy for the Logistic Regression model.
        Performs GridSearchCV for all the defined classifiers.
        Stores the cross-validation scores and the best estimators for each model.
        Creates a bar chart to visualize the cross-validation scores for each model using seaborn.

<b>Note:</b>

    The code currently only shows the training and testing accuracy for the Logistic Regression model. You can modify the code to print the results for all the models.
    This is a basic example of using GridSearchCV for hyperparameter tuning. More sophisticated techniques can be employed for better model selection.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  
from sklearn.tree import DecisionTreeClassifier  
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def cleanAndPrepareFeatures(file):
    arrest = pd.read_csv(file)
    
    ## We have to transform some columns in int (basically a mapping process since the ML algorithm needs numerical values)
    
    arrest['Arrest_Date'] = pd.to_datetime(arrest['Arrest_Date'])
    arrest['Arrest_Month'] = arrest['Arrest_Date'].dt.month
    arrest['Arrest_Day'] = arrest['Arrest_Date'].dt.day
    arrest['Arrest_Year'] = arrest['Arrest_Date'].dt.year
    
    ## I created a new column that is 1 if the day is a weekend and 0 otherwise
    arrest['Is_Weekend'] = arrest['Arrest_Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
    
    ## I selected some crimes that are basically considered as violent crimes (based on internet research)
    ## and i created a new column (for the same reason as above) that is 1 if the crime is a violent crime and 0 otherwise    
    violentCrimes = ['RAPE 1', 'RAPE 2', 'RAPE 3', 'SODOMY 1', 'STRANGULATION 1ST', 'SEXUAL ABUSE']
    arrest['Is_Violent_Crime'] = arrest['Offense_Description'].isin(violentCrimes).astype(int)
    
    ## Same stuff for the felony column
    
    arrest['Is_felony'] = (arrest['Offense_Category_Code'] == 'F').astype(int)
    
    ## We have to prepare the features for the model but we have some problems: 
    ## 1. We have to separe the categorical features and the numberic ones
    #  (like the borough, etc...) because we need numbers 
    ## 2. i double checked the dataset and i saw that there are some missing values (remove them)
    print('Null values for each category:', arrest.isnull().sum())
    
    ##1. 
    categoricalFeatures = ['Perpetrator_Race', 'Perpetrator_Sex', 'Perpetrator_Age_Group', 'Arrest_Borough', 'Arrest_Day_of_Week']
    numericalFeatures = ['Latitude', 'Longitude', 'Arrest_Month', 'Is_Weekend']
    totalFeatures = categoricalFeatures + numericalFeatures
    
    
    ##2.

    for i in totalFeatures:
        if i in arrest.columns:
            arrest = arrest.dropna(subset=[i])
            
    ## Now we have to get all the input features and the target variables
    ## X is the input features and y is the target (what we want to predict)
    X = arrest[totalFeatures]
    y = arrest['Is_Violent_Crime']
    y2 = arrest['Is_felony']
    
    ## I triedi before with preprocessing.LabelEncoder() but i didn't work this because: 
    ## the LabelEncoder() creating some order relation between the values and so give me some errors when i tird to process data 
    ## I found OneHotEncoder(), creates new columns for each category and assigns 1 or 0 to the columns
    ## https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    ## numerical data 
    
    catEncoding = OneHotEncoder()
    numScaling = StandardScaler()
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    ## THIS IS ESSENTIAL, OTHERWISE WE HAVE TO WRITE A LOT OF MORE CODE
    preprocessor = ColumnTransformer(transformers=[('cat', catEncoding, categoricalFeatures),('num', numScaling, numericalFeatures)])
    
    
    return X, y, y2, preprocessor
    

def algorihtmsApplication(X, y, preprocessor):
    
    
    ## Split the dataset in training and test set
    ## 0.3 means that we split the set 70% for traingin and 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    ## We have to create a pipeline to apply the model
    ## https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    ## PIPELINE is basically a way in which we can preprocess the data and apply the model in one shot
    ## Otherwise we should have done a lot of more steps and write a lot of more code
    
    
    print("First choise: Logistic Regression")
    
    ## WE have to apply smote because the dataset is unbalanced and we have to balance it, oterwise the model will give us a 0 precision and recall (and f1 as consequence since is the amrmonic mean between prec and recall)
    ## https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    
    smote = SMOTE(random_state=43)
    
    # logisticRegressionPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=800, random_state=42))])
    logisticRegressionPipe = ImbPipeline(steps=[('preprocessor', preprocessor),('sampling', smote),('classifier', LogisticRegression(max_iter=1000, random_state=43, class_weight='balanced'))])
    
    ## We have to train the model
    
    logisticRegressionPipe.fit(X_train, y_train)
    
    ## We have to predict the values
    
    y_pred = logisticRegressionPipe.predict(X_test)
    
    ## use all these metrics discussed during the lectures since as we know accuracy sometimes can be misleading
    
    logisticRegAccuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision) 
    print("Accuracy score: ", logisticRegAccuracy)
    
    print("\nSecond choice: K-Nearest Neighbors")
    
    ## i set n_neighbors = 5 because it's the standard and she told us it  
    
    knnPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(n_neighbors=5))])
    
    knnPipe.fit(X_train, y_train)
    
    y_pred = knnPipe.predict(X_test)
    
    knnAccuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision)    
    print("Accuracy score: ", knnAccuracy)
    
    print("\nThird choice: Naive Bayes")
    
    naiveBayesPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())])
    naiveBayesPipe.fit(X_train, y_train)
    y_pred = naiveBayesPipe.predict(X_test)
    naiveBayesPipeAccuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision) 
    print("Accuracy score: ", naiveBayesPipeAccuracy)
    
    
    print("\nFourth choice: Support Vector Machine")
    
    ## using the same pipeline as before but with the SVC, same problem 

    
    SVMPipe = ImbPipeline(steps=[('preprocessor', preprocessor),('sampling', smote),('classifier', SVC(random_state=43))])
    SVMPipe.fit(X_train, y_train)
    y_pred = SVMPipe.predict(X_test)
    SVMAccuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision) 
    print("Accuracy score: ", SVMAccuracy)
    
    ## NOT DISCUSSED IN CLASS 
    ## Random forest basically creates decision tree; each of them is trained on a random subset of the data
    ## and the result is the combination of the results of the trees (each tree make a vot to classify a data ). 
    ## The differences between a single tree and a random forest is that basically a single tree is more vulnerable to 
    ## Overfitting. Obiously having more decision tree reduce the risk of outliers (is used for really difficult problem but we can try it)
    print("\nFifth choice: Random Forest")
    
    randomForestPipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=43))])
    randomForestPipe.fit(X_train, y_train)
    y_pred = randomForestPipe.predict(X_test)
    randomForestAccuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision) 
    
    print("Accuracy score: ", randomForestAccuracy)
    
    ## NOT DISCUSSED IN CLASS
    
    print("\nSixth choice: Decision Tree")
    
    decisionTreePipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(random_state=43))])
    decisionTreePipe.fit(X_train, y_train)
    y_pred = decisionTreePipe.predict(X_test)
    decisionTreeAccuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Recall score: ", recall)
    print("F1 score: ", f1)
    print("Precision score: ", precision) 
    
    print("Accuracy score: ", decisionTreeAccuracy)
    
     
if __name__ == "__main__": 
    ## IMPORTANT:
    ## We decided to use only "Violent Crime" e "Felony" as taget variables since they are the most important ones 
    ## I've seen that these two categories are classified as violent crime and they cover the great majority of the dataset
    
    file = '../dataset/NYPD_Arrest_Data__Year_to_Date_cleaned.csv'
    
    ## We need to adjust the data to be able to use it in the model,
    ## i returned the new dataframe since we modified the original one
    print("Trying to clean the dataser and prepare the features")

    X, y, y2, preprocessor  = cleanAndPrepareFeatures(file)


    print("\nDataset cleaned :)")
    
    ## We have to apply the algorithm now, try before with Violent crime 
    print("\nApplying the algorithms for Violent Crime")
    algorihtmsApplication(X, y, preprocessor)
    print("\nApplying the algorithms for Felony")
    algorihtmsApplication(X, y2, preprocessor)
    
    
    
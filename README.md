# Customer_Purchase_Behavior
Predictive modeling to predict Spending &  Purchase using decision tree and regression modelling
This project focuses on the development of predictive models based on a sample of records from CatalogCom's dataset to  
forecast two crucial variables: (1) the likelihood of a customer making a purchase and (2) the projected spending amount   
in case of a purchase. The aim is to utilize these models to gain insights into customer behavior and identify potential  
high-value customers for targeted marketing campaigns.

**Flow of the Project:

Data Exploration and Preparation:**

Imported required libraries and functions for analysis.
Loaded the dataset into the 'df' variable using Pandas.
Examined the data structure using df.shape, revealing 2000 rows and 25 columns.
Checked for null values using df.info() and confirmed absence of nulls.
Ensured data columns were numerical (int64) and categorical variables were pre-processed as dummy variables.  

**Data Analysis - Decision Tree Model:**

Utilized Decision Tree Classification to predict "Purchase" class labels.
Split data into training and testing sets (60/40) using train_test_split from scikit-learn.
Assigned feature variables to X, excluding "Purchase" and "Spending," and set "Purchase" as the target variable (y).
Employed GridSearchCV to determine optimal hyperparameters (e.g., max depth, min sample splits, min leaf samples) for Decision Tree.
Trained the Decision Tree model using DecisionTreeClassifier() and GridSearchCV() functions.
Made predictions on the test set and evaluated model performance using a confusion matrix.  

**Model Assessment - Decision Tree:**

Analyzed the confusion matrix on the test data, identifying True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
Observed balanced performance between training and test sets, suggesting minimal overfitting.
Calculated accuracy scores for both training and test data to assess model's generalization.  

**Data Analysis - Linear Regression Model:**  

Sub-selected data where "Purchase" equals 1 to focus on purchasers.
Segregated data into training and testing sets (80/20).
Performed linear regression on the training data (X_train, y_train) to predict spending.
Predicted 'y' values for the test data using .predict() method.
Explored coefficients and predictor variables using a DataFrame table.  

**Model Assessment - Linear Regression:**

Utilized regressionSummary() from dmba library to generate a summary of regression results.
Evaluated Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Percentage Error (MPE), and Mean Absolute Percentage Error (MAPE) for both training and test data.
Assessed Mean Squared Error (MSE) and R-squared values to gauge model performance and accuracy.
Findings and Conclusions:

Noted slight overfitting in the Decision Tree model when predicting "Purchase" and more pronounced overfitting in the Linear Regression model for spending prediction.
Suggested potential enhancements, including hyperparameter tuning, cross-validation, regularization, and exploring alternative regression algorithms.
Emphasized the importance of increasing data volume and considering complexity reduction to address overfitting.
Highlighted that further analysis and refinement are necessary to improve the reliability and accuracy of the predictive models.
This project aims to provide valuable insights into customer behavior and enhance the accuracy of predictive models for effective targeted marketing strategies.

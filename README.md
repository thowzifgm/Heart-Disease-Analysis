# Heart Disease Analysis
Heart Disease Analysis - Logistic Regression, Naive Bayes, SVM, K-NN Mean, Decision Tree, Random Forest, XGBoost, Neural Network

## Introduction
Firstly, it is now more important than ever to avoid heart disorders. In order to ensure that further individuals may lead healthier lives, effective data-driven methods for forecasting cardiac problems can enhance the overall research and preventive processes. Machine learning is useful in this situation. The heart illnesses are predicted with the use of machine learning, and the forecasts are rather precise.
The study included the assessment of the patient datasets for heart disease with appropriate data computation. Then, several models were trained while predictions were produced using variety of techniques. KNN, Decision Tree, Random Forest, Support Vector Machines, and Logistic Regression are a few examples. The code and dataset I used for my Google Colab project, "Binary Classification using Sklearn and Keras," are available here. 
To forecast the existence of heart disease in a patient, I've employed a range of Machine Learning techniques that were developed in Python. The objective variable in this classification issue is a binary variable that indicates whether or not cardiac disease is present. The input features include a range of metrics.
Dataset Used: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

## Data Description
Only 14 attributes used:
#3 (age)	age
#4 (sex)	sex
#9 (cp)	chest pain type (4 values)
#10 (trestbps)	resting blood pressure
#12 (chol)	serum cholestoral in mg/dl
#16 (fbs)	fasting blood sugar > 120 mg/dl
#19 (restecg)	resting electrocardiographic results (values 0,1,2)
#32 (thalach)	maximum heart rate achieved
#38 (exang)	exercise induced angina
#40 (oldpeak)	oldpeak = ST depression induced by exercise relative to rest
#41 (slope)	the slope of the peak exercise ST segment
#44 (ca)	number of major vessels (0-3) colored by flourosopy
#51 (thal)	thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
#58 (num)	The predicted attribute
 
## Dependencies
•	numpy 
•	pandas 
•	matplotlib
•	seaborn
•	os
•	warning
•	google colab

After importing the dependencies, we verified the dataframe, viewed the shape of dataset &   printed out a few columns for verification.

## Dataset Description 
RangeIndex: 303 entries, 0 to 302

Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)

It is observed that there are no missing values.

## Columns Analysis
age:		age
sex:		1: male, 0: female
cp:		chest pain type, 1: typical angina, 2: atypical angina, 3: non anginal pain, 4: asymptomatic
trestbps:	resting blood pressure
chol:		serum cholesterol in mg/dl
fbs:		fasting blood sugar > 120 mg/dl
restecg:	resting electrocardiographic results (values 0,1,2)
thalach:	maximum heart rate achieved
exang:		exercise induced angina
oldpeak:	oldpeak = ST depression induced by exercise relative to rest
slope:		the slope of the peak exercise ST segment
ca:		number of major vessels (0-3) colored by flourosopy
thal:		thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
Target Variable Analysis
count    303.000000
mean       0.544554
std        0.498835
min        0.000000
25%        0.000000
50%        1.000000
75%        1.000000
max        1.000000
Name: target, dtype: float64

## Checking correlation between columns
target      1.000000
exang       0.436757
cp          0.433798
oldpeak     0.430696
thalach     0.421741
ca          0.391724
slope       0.345877
thal        0.344029
sex         0.280937
age         0.225439
trestbps    0.144931
restecg     0.137230
chol        0.085239
fbs         0.028046
Name: target, dtype: float64

Almost every column is well correlated with target, but 'fbs' is very weakly correlated.

## Exploratory Data Analysis
### Analyzing the Target Variable
Percentage of patience without heart problems: 45.54 Percentage of patience with heart problems: 54.46.

Analyzing 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca' and 'thal' features

We notice, that as expected, the 'sex' feature has 2 unique features. We notice, that females are more likely to have heart problems than males.
 	
As expected, the CP feature has values from 0 to 3.  We notice, that chest pain of '0', i.e., the ones with typical angina are much less likely to have heart problems.

The fbs variable does not have much significance over here. 

We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'.
 	
People with exang=1 i.e. Exercise induced angina are much less likely to have heart problems.
 	
We observe, that Slope '2' causes heart pain much more than Slope '0' and '1'.
 	
ca=4 has astonishingly large number of heart patients.
 	 

## Train Test Split
Partitioned the data into 80% training and 20% test to apply k-fold cross validation on the training data.

## Model Fitting
Logistic Regression
The accuracy score achieved using Logistic Regression is: 85.25 %.

### Naive Bayes
The accuracy score achieved using Naive Bayes is: 85.25%.

### Support Vector Machine
The accuracy score achieved using Linear SVM is: 81.97%.

### K Nearest Neighbors (KNN) Mean
The accuracy score achieved using KNN is: 67.21%.

### Decision Tree
The accuracy score achieved using Decision Tree is: 81.97%.

### Random Forest
The accuracy score achieved using Decision Tree is: 90.16%.

### XGBoost
The accuracy score achieved using XGBoost is: 85.25%.

### Neural Network
The accuracy score achieved using Neural Network is: 83.61%. Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
 
## Conclusion
Random forest has good result as compare to other algorithms. The accuracy achieved was 95%.

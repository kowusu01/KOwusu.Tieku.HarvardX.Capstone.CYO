# Classification of Income using kNN and RandomForest Algorithms

## KOwusu.Tieku.HarvardX.Capstone.CYO

![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/splash.PNG?raw=true)


## Introduction
Income disparity is common in every society and it's probably been around since man has been able to measure it. There are various reasons why income disparities exist. These include age, gender, race, level of education, etc.  

For instance, with respect to age, younger people who have just entered the workforce may earn less than their counterparts who have been in the industry for many years. The level of education may also explain income disparities since some high paying jobs often require skilled and highly educated individuals.   

Even different geographic regions or cities may offer different levels of income since some high paying jobs are sometimes more concentrated in some regions or cities than others. However, it is still difficult to measure how much these factors determine one's income.   

This project, apples kNN and randomForest to a dataset containing various attributes (predictors) and individual's income level (above or below $50K), and predict the individual's income level. 

The kNN and randomForest algorithms were chosen because they are very easy to train, and they are able handle a wide range data including a mix of categorical and numeric variables and offer good performance. 

[[read entire report...]](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/report.pdf)

## The original data

__Source__  
The data used in this project is part of the UCI machine learning datasets provided to the public. It can be found at:  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/.  

   
Three main files are provided as part of the dataset.  
  
* adult.names
* adult.data
* adult.test
 
The _adult.names_ file,  provides very useful information about the data.  The _adult.data_ and _adult.test_ files are the training and test sets. The data had already been split into training and test sets by the contributer(s). The training data contains _32561_ records. The test set contains _16281_ records, about 50% of the training set.  This provides a good amount of data for training while leaving enough for validation. Each record in the datasets represents an observation.   

The documentation indicates that the original data was retrieved from the 1994 census bureau database (http://www.census.gov/ftp/pub/DES/www/welcome.html), so the samples were taken from real dataset. The U.S. population and workforce have changed dramatically since 1994, so the patterns in this data may be different from when the samples were originally taken. 
  

__Reading the data__   
A quick peek at the data reveals that there are no column names. However, descriptions in the _adult.names_ file provides useful information in naming the columns.   
   
 
![original data - raw](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/01-raw-data.PNG?raw=true)
  

## Data distribution & trends
The following shows some of the data distributions and and general properties of the data.
![data distribution](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/data-distribution.PNG?raw=true)
  
  
![data distribution](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/income-and-education.PNG?raw=true)

  
![data distribution](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/income-and-occupation.PNG?raw=true)

  
![data distribution](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/trend-income-and-education.PNG?raw=true)


## Machine Learning

### Approach
The following approach was used the machine learning section of the project:  
0. Data partitioning (train/test)
1. Data Engineering, Transformation & Preprocessing
2. Training, Optimization & Model Selection
3. Final model and final prediction on validation set

### The caret package
In this project, I use the [caret package](https://topepo.github.io/caret/) for training the ML models.

### Data Partitioning (train/test)
The following is a breakdown of the samples used for training, testing, and final validation.
![train/test data partitions](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/train-test-data.PNG?raw=true)

### Cross validation and sampling. 
To reduce the chances of overfitting, a 5-fold cross validation was used
while training the models using a sample size of 5000 from the training set.
The following code parameters are used to define a common training control for all models:


### Performance metric
In this project, I use AUC (Area Under the Curve) as the metric for selecting best models during cross validation. In th caret training function, the metric will be set to ROC, to indicate to caret to select best model using AUC rather than Accuracy.  

When a model is fitted on test data, I use Kappa instead of accuracy when selecting best model. Kappa is often preferred to accuracy when dealing with imbalanced dataset which is the case in this project.

### Data engineering, transformation & preprocessing
The datasets used in this project had been cleaned by the original contributors. However, a few tasks were performed to prepare the data for the final analysis. These include removing/replacing special characters in categorical data, converting categorical data to dummy data, scaling and log transformation of numeric values, adding new columns, etc. The following code was used as part of the data engineering and transformation.


![code for data transformation](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/data-transformation-code.PNG?raw=true)

### Training & model tuning
The following shows an approach used in tuning the RandomForest hyper parameters to find an optimum model.

![tuning random forest hyper parameters](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/tune-rf-model.PNG?raw=true)

![chart showing random forest hyper parameter tuning](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/img-tune-rf-model.PNG?raw=true)
    
Important variables in determining income level extracted from Randomforest algorithms
The randomForest model used used in the classification ranks the variables in terms of importance. In other words, which variables have the most effect in determining whether a given income is below or above 50K.
  
The chart below shows the top 20 most important features.  From the chart, it can be seen that the capital_gain/loss variable is far the most important variable. This is followed by education_num (number of years of education), and hour_per_week.
![important variables](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/images/important-vars.PNG?raw=true)


## Conclusion
In this project, kNN and randomForest were used to predict whether an individual’s income will be <=50K
or >50K. Both algorithms performed very competitively, with randomForest winning based on Kappa performance. The final model on the validation set yielded an overall accuracy of 86%. Even with the low
prevalence of the positive class, the model is able to correctly predict the positive class 60% of the time,
which is pretty good. The model also showed that capital gain or loss is the most important variable in the
classification of whether an individual’s income is <=50K or >50K. The top four variables in the model for
predicting the outcome are capital_gain_loss, education_num, age, and hour_per_week.
The project, for the sake of time, focused on two algorithms (kNN and randomForest). Although the final
model’s performance is fairly good, an improvement may have been achieved by considering more models
and using ensemble methods. 


[[read entire report...]](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/blob/main/reports/report.pdf)


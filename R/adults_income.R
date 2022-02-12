
####################################################
##
## The main file for model analysis
##
## Running this file will initiate entire process,
## reading datafile, transforming, and analysis.
##
## Using the cross validation and number of models
## it takes about 4hrs to run from begin to end.
##
## It creates all the .rda files needed for the 
## report. 
##
###################################################

set.seed(2022)

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(wrapr)) install.packages("wrapr", repos = "http://cran.us.r-project.org")
if (!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if (!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if (!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if (!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if (!require(parallel)) install.packages("parallel", repos = "http://cran.us.r-project.org")
if (!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if (!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")

library(matrixStats)
library(dplyr)
library(ggplot2)
library(readr)
library(caret)
library(randomForest)
library(broom)
library(RColorBrewer)
library(scales)
library(reshape2)
library(parallel)
library(doParallel)
library(lattice)
library(wrapr)      # for functions such as qc()


# url to the original source
TRAINING_DATA.URL   <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
VALIDATION_DATA.URL <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# for informational purposes
INFO.URL            <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"


# url to my github location, just in case
TRAINING_DATA.GITHUB.URL <- "https://raw.githubusercontent.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/tree/main/data/adult.data"
VALIDATION_DATA.GITHUB.URL <- "https://raw.githubusercontent.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.CYO/tree/main/data/adult.test"


TRAIN_DATA_PATH <- "data/adult.data"
VALIDATION_DATA_PATH  <- "data/adult.test"

#used for cross validation and training
SAMPLE.SIZE <- 5000
CV.NUM <- 5
TUNE.LENGTH <- 10


# set up parallel processing for model training
num_cores <- makeCluster(detectCores() - 1)
registerDoParallel(cores = num_cores)

# attempt download files from source
#if(!file.exists(TRAIN_DATA_PATH))
#  download.file(TRAINING_DATA.URL,   TRAIN_DATA_PATH,  method="curl", mode = "w")

#if(!file.exists(VALIDATION_DATA_PATH))
#  download.file(VALIDATION_DATA.URL, VALIDATION_DATA_PATH,  method="curl", mode = "w")


#################################################################################
# attempt download files from my github repo
################################################################################
if(!file.exists(TRAIN_DATA_PATH))
  download.file(TRAINING_DATA.GITHUB.URL,   TRAIN_DATA_PATH,  method="curl", mode = "w")

if(!file.exists(VALIDATION_DATA_PATH))
  download.file(VALIDATION_DATA.GITHUB.URL, VALIDATION_DATA_PATH,  method="curl", mode = "w")
################################################################################


################################################################################
# read the data files into memory
################################################################################


# use wrapr::qc() function to create a list of quoted string for the column headers
COL_HEADERS <- qc(age, workclass, final_weight, education, education_num,
                  marital_status, occupation, relationship, race, sex,
                  capital_gain, capital_loss, hours_per_week, native_country,
                  Income)

# read training and test datasets into memory
adult_income_train <- read.csv(TRAIN_DATA_PATH, col.names = COL_HEADERS)
adult_income_validation  <- read.csv(VALIDATION_DATA_PATH, col.names = COL_HEADERS, skip = 1)

# trim all character columns
adult_income_train <- adult_income_train %>% mutate(across (where(is.character), str_trim))
adult_income_validation <- adult_income_validation %>% mutate(across (where(is.character), str_trim))

saveRDS(adult_income_train, "rda/adult_income_train.rda")
saveRDS(adult_income_validation, "rda/adult_income_validation.rda")



#####################################################################
## DATA QUALITY CHECKS
#####################################################################

# 1. check that correct number of rows was read
TRAIN_NUM_OF_ROWS_EXPECTED <- 32561
VALIDATION_NUM_OF_ROWS_EXPECTED  <- 16281

verify_stats <- data.frame("Item" = c("train_num_rows", "validation_num_rows"))
verify_stats$Expected <- c(TRAIN_NUM_OF_ROWS_EXPECTED, VALIDATION_NUM_OF_ROWS_EXPECTED)
verify_stats$Actual   <- c(nrow(adult_income_train), nrow(adult_income_validation))

#2. check for NA in all columns
verify_stats$HasNA   <- c(any(is.na(adult_income_train)), any(is.na(adult_income_validation)))


#3. numeric columns: check range of values, min/max

# function to check min/max
fnFindMinMax <- function(myFeature){
  data.frame("min" = min(myFeature), "max" = max(myFeature))
}
sapply( adult_income_train %>% select(where(is.numeric)), fnFindMinMax)

#4. for categorical values, inspect the unique values
fnShowUniqueValues <- function(myFeature){
  unique(myFeature)
}

#adult_income_train %>% select(where(is.character)) %>% sapply(fnShowUniqueValues)


###########################################
## data engineering 
##########################################


################################################################################
# function create a new column based on capital_gain and capital_loss
# 
################################################################################
fnCreateNewCapitalGainLossColumn <- function(my_dataset){
  
  # combine the the capital gain/loss variables into one
  # all capital losses become negative
  my_dataset %>% 
    mutate(capital_gain_loss = capital_gain - capital_loss,
           gain_or_loss = if_else(capital_gain>0, "gain", if_else(capital_loss>0, "loss", "none")))
  
}


fnRenameSomeCategoricalValues <- function(my_dataset){
  #1. update some categorical values
  #   remove upadte all ? marks values to Unspecified
  my_dataset <- my_dataset %>% 
    mutate(workclass = if_else(workclass=="?", "Unspecified", workclass))
  
  my_dataset <- my_dataset %>% 
    mutate(occupation = if_else(occupation=="?", "Unspecified", occupation))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="?", "Unspecified", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="United-States", "UnitedStates", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="Puerto-Rico", "PuertoRico", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="Dominican-Republic", "DominicanRepublic", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="El-Salvador", "ElSalvador", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="Outlying-US(Guam-USVI-etc)", "Outlying_US_Guam_USVI", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="Trinadad&Tobago", "TrinadadTobago", native_country))
  
  my_dataset <- my_dataset %>% 
    mutate(native_country = if_else(native_country=="Holand-Netherlands", "HolandNetherlands", native_country))
  
  # replace all hyphens in categorical values - this will save a lot of headaches later
  my_dataset <- my_dataset %>% mutate(across(where(is.character), ~ str_replace_all(.x, "-", "_")))
  
  my_dataset
  
}


###############################################################################
# manual transformations
###############################################################################
fnManualTransformations <- function(my_dataset){
  
  # 1. add the column created for capital gain/loss, we will use it in the analysis
  my_dataset <- fnCreateNewCapitalGainLossColumn(my_dataset)
  my_dataset <- my_dataset %>% select(-c("gain_or_loss"))
  
  # 2. rename some categorical values, e.g. ? becomes "Unspecified", etc
  my_dataset <- fnRenameSomeCategoricalValues(my_dataset)
  
  # change all character features to factor
  my_dataset <- my_dataset %>% mutate_if(is.character, factor)
  
  my_dataset <- my_dataset %>% mutate(Income=factor(Income))
  my_dataset$Income <- relevel(my_dataset$Income, ref = ">50K")
  
  # 1. education_num, hours_per_week - center and scale - leave for caret to take care of
  
  # 2. age, final_weight - log transformation before caret preprocess
  my_dataset <-  my_dataset %>% mutate(across(c("age", "final_weight"), log2 ))
  
  # 3. capital_gain/loss - adjust, and log transform before caret preprocess
  my_dataset <- my_dataset %>%
    mutate(capital_gain_loss = capital_gain_loss + max(capital_loss) + 1 )
  
  my_dataset <- my_dataset %>%
    mutate(capital_gain_loss = log2(capital_gain_loss) )
  
  # now remove the original capital gain/loss vars
  my_dataset <- my_dataset %>% select(-c("capital_gain", "capital_loss"))
  
  my_dataset
}


fnCreateCaretDummyVarsPreproc <- function(my_dataset) {
  
  # for some reason, caret wants the response var to be an int
  my_dataset <- my_dataset %>% mutate(Income = as.integer(Income))
  
  # create the preprocessed object that can be applies to other datasets like test, and validation
  caret_dummy_vars_preprocessor  <- dummyVars(Income ~ ., data = my_dataset)
  
  # return  preprocessor for future use
  caret_dummy_vars_preprocessor
}

fnApplyCaretDummyVarsPreproc <- function(my_dataset, caret_dummy_vars_preprocessor) {
  
  income_col <- my_dataset %>% select("Income")
  
  my_dataset <-  predict(caret_dummy_vars_preprocessor, newdata = my_dataset) %>% as.data.frame()
  
  # the final result from predict does not include the response var, add it back
  income_col %>% cbind(my_dataset)
}


fnCreateCaretScalePreproc <- function(my_dataset){
  # scale everything except the response var
  my_dataset[, -1] %>% preProcess(method = c("center", "scale"))
}

fnApplyCaretScalePreproc <- function(my_dataset, caret_scale_preprocessor){
  response_col <- my_dataset %>% select("over50K")
  scaled_features <- predict(caret_scale_preprocessor, my_dataset[, -1]) %>% as.data.frame()
  cbind(response_col, scaled_features)
}

fnEncodeResponseVar <- function(my_dataset){
  my_dataset <- my_dataset %>% mutate(Income=if_else(Income==">50K", "Y", "N"))
  my_dataset <- my_dataset %>% mutate(Income=factor(Income))
  my_dataset$Income <- relevel(my_dataset$Income, ref = "Y")
  
  # rename the predicted class (Income) to over50K
  names(my_dataset)[1] <- "over50K"
  
  my_dataset
  
}

##############################################################################
# functions end
##############################################################################



# make a copy of the original data and lets modify it for visualization
adult_income <- adult_income_train


###############################################################################
# transform some of the variables for visualization purposes
###############################################################################


adult_income <- fnCreateNewCapitalGainLossColumn(adult_income)


## work class
adult_income <- adult_income %>% 
  mutate(workclass_collapsed = fct_collapse(workclass,
                                            "Unspecified" = "?",
                                            "Self_employed"    = c("Self-emp-inc", "Self-emp-not-inc"),
                                            "Private"          = "Private",
                                            "State_local_gov"  = c("State-gov", "Local-gov"), 
                                            "Federal_gov"      = "Federal-gov" ))




###############################################################################
# transform some of the variables for visualization purposes
###############################################################################

education_levels <- c("Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
                      "Some-college", "Assoc-acdm", "Assoc-voc", "Bachelors", "Masters","Prof-school", "Doctorate")

adult_income$education <- factor(adult_income$education, levels = education_levels)
adult_income <- adult_income %>% mutate_if(is.character, factor)




## add age_group column
adult_income <- adult_income %>% mutate(age_group = if_else(age<=20,"<=20", 
                                                            if_else(age<=25, "<=25",
                                                                    if_else(age<=30, "<=30",
                                                                            if_else(age<=35, "<=35",
                                                                                    if_else(age<=40, "<=40",
                                                                                            if_else(age<=45, "<=45",
                                                                                                    if_else(age<=50, "<=50",
                                                                                                            if_else(age<=55, "<=55",
                                                                                                                    if_else(age<=60, "<=60",
                                                                                                                            if_else(age<=65, "<=65","65+"))))))))))) 


## work class
adult_income <- adult_income %>% 
  mutate(workclass_collapsed = fct_collapse(workclass,
                                            "Unspecified" = "?",
                                            "Self_employed"    = c("Self-emp-inc", "Self-emp-not-inc"),
                                            "Private"          = "Private",
                                            "State_local_gov"  = c("State-gov", "Local-gov"), 
                                            "Federal_gov"      = "Federal-gov" ))



###############################################################
# education: 
# 
# - combine all levels before some college into one category
# - also combine Doctorate and Professional degrees into one category
#
# create a more general (collapsed) categories of education
adult_income <- adult_income %>% mutate(education_collapsed = fct_collapse(education,
                                                                           No_college_degree = c("Preschool", "1st-4th", "5th-6th", "7th-8th","9th", "10th",  "11th",  "12th", "HS-grad", "Some-college"),
                                                                           Associate_degree = c("Assoc-acdm","Assoc-voc"),
                                                                           Bachelors = "Bachelors",
                                                                           Graduate_post_graduate = c("Masters", "Prof-school", "Doctorate")))



#######################################################################
## Occupation
## Military Specific Occupations - excluded since it has only  9 records
#adult_income <- adult_income %>% filter(occupation != "Armed-Forces")
adult_income <- adult_income %>% mutate(occupation_collapsed = fct_collapse(occupation,
                                                                            Unspecified = "?",
                                                                            Management_business_science_and_arts = c("Exec-managerial","Prof-specialty"),
                                                                            Service = c("Protective-serv", "Priv-house-serv", "Handlers-cleaners", "Other-service"),
                                                                            Sales_and_Office = c("Sales","Adm-clerical", "Tech-support"),
                                                                            Natural_resources_construction_and_maintenance = c("Farming-fishing", "Craft-repair", "Machine-op-inspct"),
                                                                            Production_transportation_and_material_moving = "Transport-moving",
))




adult_income <- adult_income %>% mutate(capital_gain_level = if_else(capital_gain==0,"none", 
                                                                     if_else(capital_gain<=10000, "<=10K",
                                                                             if_else(capital_gain<=20000, "<=20K",">20K"))))

adult_income <- adult_income %>% mutate(capital_loss_level = if_else(capital_loss==0,"none", 
                                                                     if_else(capital_loss<=1000, "<=1K",
                                                                             if_else(capital_loss<=2000, "<=2K",
                                                                                     if_else(capital_loss<=3000, "<=3K",">3K")))))

adult_income <- adult_income %>% mutate(capital_gain_level = factor(capital_gain_level, levels = c("none", "<=10K", "<=20K", ">20K")))
adult_income <- adult_income %>% mutate(capital_loss_level = factor(capital_loss_level, levels = c("none", "<=1K", "<=2K", "<=3K", ">3K")))


#adult_income <- fnCreateNewCapitalGainLossColumn(adult_income)
saveRDS(adult_income, "rda/adult_income.rda")




################################################################################
#
# prepare the data before for the actual preprocessing by caret
#
################################################################################

#########################################################################
# prepare original training dataset to be used for the models:
# - all categorical predictors encoded
#
# resource: https://www.r-bloggers.com/2020/02/a-guide-to-encoding-categorical-features-using-r/
#
#########################################################################

# 0. partition the training dataset into train/holdout(test).
# The original test set will be used for validation

test_index <- createDataPartition(y = adult_income_train$Income, times = 1, p = 0.1, list = FALSE)
train_set <- adult_income_train[-test_index,]
test_set  <- adult_income_train[test_index,]

saveRDS(test_set, "rda/test_set.rda")


###############################################################################
#
# PREPROCESSING PIPELINE
# TAKE ALL DATASETS THROUGH THE SAME TRANSFORMATION
#
###############################################################################
# 1. rename some categorical values

train_set <- train_set %>% select(-c("education"))
training_set_final <- fnManualTransformations(train_set)
test_set_final     <- fnManualTransformations(test_set)

# 2. create caret preprocessor for dummy variables
caret_dummy_vars_preprocessor <- fnCreateCaretDummyVarsPreproc(training_set_final)
training_set_final <- fnApplyCaretDummyVarsPreproc(training_set_final,caret_dummy_vars_preprocessor)
test_set_final     <- fnApplyCaretDummyVarsPreproc(test_set_final,caret_dummy_vars_preprocessor)

# 3. cosmetic stuff
# convert response var to Y,N, and set the positive class to Y (over50K)
training_set_final <- fnEncodeResponseVar(training_set_final)
test_set_final     <- fnEncodeResponseVar(test_set_final)

# 4. perform final caret preprocess - center and scale
caret_scale_preprocessor <- fnCreateCaretScalePreproc(training_set_final)
training_set_final <-  fnApplyCaretScalePreproc(training_set_final, caret_scale_preprocessor)
test_set_final     <-  fnApplyCaretScalePreproc(test_set_final, caret_scale_preprocessor)



################################################################################
# perform the same transformation on validation set
################################################################################

# 1. rename some categorical values
#adult_income_validation <- readRDS("rda/adult_income_validation.rda")
adult_income_validation <- adult_income_validation %>% mutate_at(c("Income"), ~ str_replace(.x, "\\.", ""))

adult_income_validation     <- fnManualTransformations(adult_income_validation)

# 2. create caret preprocessor for dummy variables
adult_income_validation     <- fnApplyCaretDummyVarsPreproc(adult_income_validation,caret_dummy_vars_preprocessor)

# 3. cosmetic stuff
adult_income_validation     <- fnEncodeResponseVar(adult_income_validation)

# 4. perform final caret preprocess - center and scale
adult_income_validation     <-  fnApplyCaretScalePreproc(adult_income_validation, caret_scale_preprocessor)

glimpse(adult_income_validation)

saveRDS(training_set_final, "rda/training_set_final.rda") 
saveRDS(test_set_final, "rda/test_set_final.rda") 
saveRDS(adult_income_validation, "rda/adult_income_validation_final.rda") 

################################################################################
## END PREPROCESSING
################################################################################


################################################################################
# set up sample size to use for training
################################################################################

index <- sample(nrow(training_set_final), SAMPLE.SIZE)
train_sample <- training_set_final[index, ]

train_x <- training_set_final %>% select(-c("over50K")) %>% as.matrix()
test_x <- test_set_final %>% select(-c("over50K")) %>% as.matrix()

train_y <- training_set_final[, "over50K"]
test_y <- test_set_final[, "over50K"] 



################################################################################
## MODEL SUPPORT FUNCTIONS
################################################################################


cv_control <- trainControl(method = "cv", number = CV.NUM, summaryFunction=twoClassSummary, classProbs=TRUE, savePredictions = T)

# model 1 - use caret defaults to train rf model using cross validation
# use ROC as a metric to for selecting best model
fnTrainRandomforestWithDefaults <- function(train_sample){
  train(
    over50K ~ .,
    method = "rf", 
    metric="ROC",
    data = train_sample,
    trControl = cv_control,
    tuneLength=TUNE.LENGTH,
    allowParallel=TRUE
  )
}

# model 2 - train an rf model with cross validation and with different values for best mtry
fnTrainRandomforest_Tune_mtry <- function(train_sample){
  train(
    over50K ~ .,
    method = "rf", 
    data = train_sample,
    metric="ROC",
    trControl = cv_control,
    tuneGrid  = data.frame(mtry = c(5, 10, 15, 20, 30)),
    allowParallel=TRUE
  )
}

# for training models 3, 4 to find the best nodesize
# given a fixed mtry value, train an rf model and with different values for nodesize
fnTrainRandomforest_Tune_nodesize <- function(train_sample, selected_mtry){
  
  nodesize <- c(seq(10, 100, 5),150,200)
  map_df(nodesize, function(n){
    
    rf_train <- 
      train(
        over50K ~ .,
        method = "rf", 
        data = train_sample,
        tuneGrid  = data.frame(mtry = c(selected_mtry)),
        allowParallel=TRUE,
        nodesize = n)
    
    list(nodesize=n,accuracy=rf_train$results$Accuracy, kappa=rf_train$results$Kappa)
    
  } )
}

#######################################################
## function to extract relevant information from
#  confusion matrix
#######################################################
fnTidyCMStats <- function(cm)
{
  
  tidy_model_stats <- tidy(cm)
  tidy_model_stats <- tidy_model_stats %>% select(-c("class"))
  
  tidy_model_stats <- tidy_model_stats %>% filter(term != "mcnemar") %>% select(term, estimate) 
  
  extra_info <- data.frame(   
    term=c("accuracy.conf.low", 
           "accuracy.conf.high", 
           "no_information_rate"),
    estimate=c( cm$overall['AccuracyLower'], 
                cm$overall["AccuracyUpper"], 
                cm$overall["AccuracyNull"])
  )
  
  row.names(extra_info) <- NULL
  rbind(tidy_model_stats, extra_info)
  
}


# given rf model, fit it on entire training and test sets, 
# and make predictions
fnPredictOnTest <- function(rf_fit, train_set, test_set, model_name){
  
  pred_test   <- predict(rf_fit, test_set, type="class")
  confusionMatrix_test  <- confusionMatrix(pred_test,  test_set[, "over50K"] )
  
  saveRDS(confusionMatrix_test, paste0("rda/", model_name, "_confusionMatrix.rda"))
  saveRDS(fnTidyCMStats(confusionMatrix_test), paste0("rda/", model_name, "_confusionMatrix_tidy.rda"))
  saveRDS(pred_test, paste0("rda/", model_name, "_predictions.rda"))
  
  data.frame(test.accuracy = confusionMatrix_test$overall["Accuracy"],
             test.kappa = confusionMatrix_test$overall["Kappa"],
             test.sensitivity = confusionMatrix_test$byClass["Sensitivity"], 
             test.specificity = confusionMatrix_test$byClass["Specificity"]  )
  
}


######################################################
# knn model training
######################################################

print(paste("training knn model 1 - ", Sys.time()) )
knn_model_1 <- train(
  over50K ~ .,
  method = "knn",
  metric="ROC",
  data = train_sample,
  trControl = cv_control,
  tuneLength=TUNE.LENGTH
)
print(paste("done training knn model 1 - ", Sys.time()) )


knn_k_tuning_grid   <- data.frame(k = c(seq(1, 39, 2), 50, 100, 150, 200, 250, 350))
print(paste("starting knn model 2 ... ", Sys.time()) )
knn_model_2 <- train(
  over50K ~ .,
  method = "knn",
  metric="ROC",
  data = train_sample,
  trControl = cv_control,
  tuneGrid = knn_k_tuning_grid
)
print(paste("done training knn model 2 - ", Sys.time()) )
#knn_model_1
#knn_model_2

saveRDS(knn_model_1$results, "rda/knn_model_1_cv_results.rda")
saveRDS(knn_model_2$results, "rda/knn_model_2_cv_results.rda")

plot(knn_model_1)
dev.new()
plot(knn_model_2)

# knn fit on entire training set
print(paste("fitting knn model on entire training set - ", Sys.time()))
knn_fit_1 <- knn3(over50K ~ ., data=training_set_final, k=knn_model_1$bestTune$k)
knn_fit_2 <- knn3(over50K ~ ., data=training_set_final, k=knn_model_2$bestTune$k)
print(paste("done fitting knn model on entire training set ", Sys.time()) )

knn_pred_1 <- predict(knn_fit_1, newdata=test_set_final, type="class")
knn_pred_2 <- predict(knn_fit_2, newdata=test_set_final, type="class")

saveRDS(knn_pred_1, "rda/knn_model_1_predictions.rda")
saveRDS(knn_pred_2, "rda/knn_model_2_predictions.rda")

knn_confusionMatrix_1 <- confusionMatrix(knn_pred_1, test_set_final$over50K)
knn_confusionMatrix_2 <- confusionMatrix(knn_pred_2, test_set_final$over50K)
saveRDS(knn_confusionMatrix_1, "rda/knn_model_1_confusionMatrix.rda")
saveRDS(knn_confusionMatrix_2, "rda/knn_model_2_confusionMatrix.rda")

saveRDS(fnTidyCMStats(knn_confusionMatrix_1), "rda/knn_model_1_confusionMatrix_tidy.rda")
saveRDS(fnTidyCMStats(knn_confusionMatrix_2), "rda/knn_model_2_confusionMatrix_tidy.rda")
print(paste("knn done. - ", Sys.time()))


######################################################
# END knn model training
######################################################



######################################################
# rf model training
######################################################

print(paste("training rf model - ", Sys.time()) )

# ##############################################################
# 1. user caret to perform cross validation, using caret defaults

model_1 <- fnTrainRandomforestWithDefaults(train_sample)
print ("done training model_1")
print (paste0("fitting model_1 with best mtry=", model_1$bestTune$mtry, " - ", Sys.time()))
fit_1 <- randomForest(over50K ~ ., data = training_set_final, mtry = model_1$bestTune$mtry, importance = TRUE)
Sys.time()
(performance_stats_1 <- fnPredictOnTest(fit_1, training_set_final, test_set_final, "rf_model_1"))
dev.new()
plot(model_1)
saveRDS(model_1, "rda/rf_model_1.rda")
saveRDS(model_1$results, "rda/rf_model_1_results.rda")
Sys.time()


# ##############################################################
# 2. based the tuning parameters caret used in the cv in step 2, attempt to optimize mtry
Sys.time()
model_2 <- fnTrainRandomforest_Tune_mtry(train_sample)
print ("done training model_2")
print (paste0("fitting model_2 with best mtry=", model_2$bestTune$mtry, " - ", Sys.time()))

fit_2  <- randomForest(over50K ~ ., data = training_set_final, mtry = model_2$bestTune$mtry, importance = TRUE)
(performance_stats_2 <- fnPredictOnTest(fit_2, training_set_final, test_set_final, "rf_model_2"))
saveRDS(model_2, "rda/rf_model_2.rda")
saveRDS(model_2$results, "rda/rf_model_2_results.rda")
Sys.time()


# ##############################################################
# 3. setting mtry constant to the value obtained from step 3, 
#    attempt to tune nodesize
Sys.time()
model_3 <- fnTrainRandomforest_Tune_nodesize(train_sample, model_2$bestTune$mtry)
(model_3_best_nodesize <- model_3[which.max(model_3$kappa), 1] %>% pull(nodesize))

print ("done training model_3")
print (paste0("fitting model_3 with best mtry=", model_2$bestTune$mtry, ", nodesize=", model_3_best_nodesize, " - ", Sys.time()))
fit_3  <- randomForest(over50K ~ ., data = training_set_final, 
                       mtry = model_2$bestTune$mtry, 
                       nodesize=model_3_best_nodesize, importance = TRUE)

(performance_stats_3 <- fnPredictOnTest(fit_3, training_set_final, test_set_final, "rf_model_3"))
saveRDS(model_3, "rda/rf_model_3.rda")
saveRDS(performance_stats_3, "rda/rf_model_3_results.rda")
Sys.time()

# add the best model from knn to our list of performance statistics
best_knn_model <- data.frame( 
                           test.accuracy = knn_confusionMatrix_1$overall["Accuracy"], 
                           test.kappa = knn_confusionMatrix_1$overall["Kappa"],
                           test.sensitivity = knn_confusionMatrix_1$byClass["Sensitivity"], 
                           test.specificity = knn_confusionMatrix_1$byClass["Specificity"])

perf_stats <- performance_stats_1 %>% 
    rbind(best_knn_model) %>% 
    rbind(performance_stats_2) %>% 
    rbind(performance_stats_3)


perf_stats

print(paste("done training rf model - ", Sys.time()) )

row.names(perf_stats) <- c("knn_best_model", "rf_model.1 (using caret defaults)", "rf_model.2 (find best mtry)", "rf_model.3 (find best nodesize)")

saveRDS(perf_stats, "rda/rf_test_performance_stats.rda")


#####################################################################
# HOORAY! FINAL PREDICTIONS
#####################################################################

predict_final <- predict(fit_3, adult_income_validation, type="class")
confusionMatrix_final  <- confusionMatrix(predict_final,  adult_income_validation[, "over50K"] )

saveRDS(predict_final, "rda/predictions_final.rda")
saveRDS(confusionMatrix_final, "rda/confusionMatrix_final.rda")

print(paste( "done fitting final model on validation set - ", Sys.time()))

#####################################################################

#####################################################################
# VARIABLE IMPORTANCE
#####################################################################

feature_importance <- importance(fit_3, type = 1) %>%  as.data.frame() %>% arrange(desc(MeanDecreaseAccuracy))
row_names <- row.names(feature_importance)
row.names(feature_importance) <- NULL
feature_importance <- data.frame(Feature=row_names) %>% cbind(feature_importance) %>% mutate(Feature=factor(Feature, levels = Feature))
saveRDS(feature_importance, "rda/important_features.rda")

print(paste("DONE!! - ", Sys.time()) )

#####################################################################
# HOORAY! THE END
#####################################################################

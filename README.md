# MachineLearning
Course project work

Remove columns with little/no data.

·         Create Training and test data from training data for cross validation checking

·         Use Random Forest to train the model and predict the output for quiz

library(ggplot2)
library(caret)
## Loading required package: lattice
library(randomForest)
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
library(e1071)
library(gbm)
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
library(doParallel)
## Loading required package: foreach
## Loading required package: iterators
library(survival)
library(splines)
library(plyr)
setwd("~/GitHub/PracMacLearn")
Load data

Load data.
Remove “#DIV/0!”, replace with an NA value.
# load data
training <- read.csv("~/GitHub/PracMacLearn/data/pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("~/GitHub/PracMacLearn/data/pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
training <- training[, 6:dim(training)[2]]
 
treshold <- dim(training)[1] * 0.95
#Remove columns with more than 95% of NA or "" values
goodColumns <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)
 
training <- training[, goodColumns]
 
badColumns <- nearZeroVar(training, saveMetrics = TRUE)
 
training <- training[, badColumns$nzv==FALSE]
 
training$classe = factor(training$classe)
 
#Partition rows into training and crossvalidation
inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]
 
 
testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodColumns]
testing$classe <- NA
testing <- testing[, badColumns$nzv==FALSE]
#Train model
mod <- train(classe ~ ., data=training, method="rf")
 
pred <- predict(mod, crossv)
#show confusion matrices
confusionMatrix(pred, crossv$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    1 1135    6    0    0
##          C    0    1 1020    4    0
##          D    0    0    0  960    1
##          E    1    0    0    1 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.996    0.994    0.995    0.999
## Specificity             0.999    0.999    0.999    1.000    1.000
## Pos Pred Value          0.998    0.994    0.995    0.999    0.998
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.184
## Detection Prevalence    0.285    0.194    0.174    0.163    0.184
## Balanced Accuracy       0.999    0.998    0.997    0.997    0.999
 
 
 
 
#out-of-sample error
pred <- predict(mod, crossv_test)
accuracy <- sum(pred == crossv_test$classe) / length(pred)
Based on results, the Random Forest prediction was very good. The RF model will be used as the sole prediction model. The confusion matrix created gives an accuracy of 99.6%. This is excellent.

As a double check the out of sample error was calculated. This model achieved 99.7449 % accuracy on the validation set.

Conclusion

I stopped at this stage as the goal to be able to get the required answers and report the errors achieved with the model has been reached without any further fine tuning.

The Random Forest method worked very well.

The Confusion Matrix achieved 99.6% accuracy. The Out of Sample Error achieved 99.7449 %.

This model will be used for the final calculations.

The logic behind using the random forest method as the predictor rather than other methods or a combination of various methods is:

 
o   Random forests are suitable when to handling a large number of inputs, especially when the interactions between variables are unknown.
 
o   Random forest’s built in cross-validation component that gives an unbiased estimate of the forest’s out-of-sample (or bag) (OOB) error rate.
 
o   A Random forest can handle unscaled variables and categorical variables. This is more forgiving with the cleaning of the data.
 
o   The down side of this method is that it takes quite much time to train the model. For it took about 30 min to train the model on Intel i5 8 GB RAM
 
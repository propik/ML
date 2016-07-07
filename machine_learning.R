library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(doParallel)
library(survival)
library(splines)
library(plyr)
training <- read.csv("C:/Users/dmitrijs.finaskins/Downloads/pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("C:/Users/dmitrijs.finaskins/Downloads/pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
training <- training[, 6:dim(training)[2]]
treshold <- dim(training)[1] * 0.95

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

mod <- train(classe ~ ., data=training, method="rf")
pred<- predict(mod, crossv)
confusionMatrix(pred, crossv$classe)
# PRACTICAL MACHINE LEARNING - WEIGHT LIFTING ASSESSMENTS
Todd Sobiech  
Sunday, January 31, 2016  



####Project Description
Six test participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
Data was collected from accelerometers on the belt, forearm, arm, and dumbell of each participant. 

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell
Biceps Curl in five different fashions: exactly according to the specification 
(Class A)  exactly according to the specification
(Class B)  throwing the elbows to the front 
(Class C)  lifting the dumbbell only halfway 
(Class D)  lowering the dumbbell only halfway 
(Class E)  throwing the hips to the front 

More information is available from the website: http://groupware.les.inf.puc-rio.br/har



####Project Goal
Predict how they complated the exercise from 5 possible outcomes(classes)



####Data Overview
There are two datasets.

Training Data - consisting of 19,622 records and 160 variables
Testing Data  - consiting of 20 records and 160 variables

The target variable is defined as "Classe" within the training data


####STEP 1 - LOAD THE DATA
The data for this analysis was loaded from the following locations:
Training - http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
Testing  - http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
rm(list = ls())
if (!file.exists("pml-training.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
}
```

####Step 2 - Process the data
Assign the datasets to a dataframe.  Set blank records to "NA"


```r
traindata <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA"))
testdata <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA")) 
```


####Step 3 - Examine the data


```r
dim(traindata)  #19622, 160
```

```
## [1] 19622   160
```

```r
summary(traindata$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
#summary(traindata)
dim(testdata)  #20, 160
```

```
## [1]  20 160
```

```r
#str(testdata)
```

There are 19,622 records in the training data.  There are 160 variables and many do not contain data.
The variables missing data do not provide value to the prediction exercise in this case and should be removed.
The values of the classifications are pretty evenly split across classes (B-E) while class A has the highest value.


####Step 4 - Remove the variables containing NO data
There are 159 variables available to use for building the prediction model.  However, many of these variables
do not contain any data and are going to be removed.


```r
na_traindata = sapply(traindata, function(x) {sum(is.na(x))})
#na_traindata
table(na_traindata)  #19216
```

```
## na_traindata
##     0 19216 
##    60   100
```


There are 100 variables with almost all missing values. These variables were removed



```r
missing_columns = names(na_traindata[na_traindata==19216])
traindata2 = traindata[, !names(traindata) %in% missing_columns]
#str(traindata2)
dim(traindata2)  #19622,60
```

```
## [1] 19622    60
```


After removing the variables with no data, 60 variables are remaining in the training data for use
for analysis


####Step 5 - Prepare the R Instance 
Ensure all packages that will be needed for the analysis are referenced


```r
Sys.info()[1:2]
```

```
##   sysname   release 
## "Windows"   "7 x64"
```

```r
R.version.string
```

```
## [1] "R version 3.2.2 (2015-08-14)"
```

```r
getwd()
```

```
## [1] "C:/Users/Todd/AppData/Local/Temp/RtmpUd6Oou"
```

```r
#install.packages('caret', dependencies = TRUE)
#install.packages("AppliedPredictiveModeling")
#install.packages("rattle")
#install.packages("rpart.plot")
#install.packages("randomForest")
#install.packages("ggthemes")
#install.packages("gridExtra")
#install.packages("ggplot2")
#install.packages("grid")

library(caret)
library(AppliedPredictiveModeling)
library(rattle)
library(rpart.plot)
library(randomForest)
library(ggthemes)
library(gridExtra)
library(ggplot2)
library(grid)
```



####Step 6 - Partition the Data
Seperate the training dataset into two groups.  One group consisting of 75% of the records to be
used to build the model and 25% of the records to test the model as assess fit.


```r
set.seed(1234)
inTrain = createDataPartition(traindata2$classe, p = 0.75, list = F)
#inTrain
training = traindata2[inTrain,]
testing = traindata2[-inTrain,]
```


The first 7 columns of the training data was removed because these variables do not provide relevant
content to aid in building the model



```r
training = training[,-c(1:7)]
#str(training)
dim(training)
```

```
## [1] 14718    53
```

```r
dim(testing)
```

```
## [1] 4904   60
```

The training group contains 14,718 records and 53 variables.
The testing groupp contains 4,904 variables.



####Step 7 - Identify Important Variables
Using random forest, the most important variables related to the target variable "Classe" are identified.


```r
#fsRF = randomForest(training[,-outcome], training[,outcome], importance = T)
fsRF = randomForest(training, training$classe, importance = T)
#fsRF
rfImp = data.frame(fsRF$importance)
#head(rfImp[order(rfImp$MeanDecreaseGini,decreasing=TRUE), ],20)
rfImp2<-head(rfImp[order(rfImp$MeanDecreaseGini,decreasing=TRUE), ],20)
rfImp2[c(7)]
```

```
##                   MeanDecreaseGini
## classe                   4582.7128
## roll_belt                 684.0458
## yaw_belt                  394.3658
## pitch_forearm             363.0251
## pitch_belt                330.3904
## magnet_dumbbell_z         312.3178
## magnet_dumbbell_y         306.3109
## roll_forearm              235.4565
## accel_belt_z              233.4844
## magnet_belt_y             223.7002
## magnet_dumbbell_x         223.2944
## magnet_belt_z             216.1197
## roll_dumbbell             208.6216
## accel_dumbbell_y          206.8489
## accel_forearm_x           162.8930
## gyros_belt_z              153.2137
## accel_dumbbell_z          141.7138
## total_accel_belt          138.0744
## roll_arm                  126.7119
## accel_dumbbell_x          123.3533
```

```r
#getTree(fsRF,1)
```

Based on the Mean Dcrease of the Gini, the roll_belt, yaw_belt, pitch_forearm, pitch_belt and 
magnet_Dumbbell_y are the most important variables.


####Step 8 - Construct Predictive Models

Two models were selected to fit to the training data.  A forest and trees model using 3 fold cross validation and a decision tree using the rpart function


```r
#Nearest Neighbor
#ctrlKNN = trainControl(method = "adaptive_cv")
#modelKNN = train(classe ~ ., training, method = "knn", trControl = ctrlKNN)
#resultsKNN = data.frame(modelKNN$results)

#Random Forest
#ctrlRF = trainControl(method = "oob")
#modelRF = train(classe ~ ., training, method = "rf", ntree = 200, trControl = ctrlRF)
#resultsRF = data.frame(modelRF$results)

#Random Forest 2
# instruct train to use 3-fold CV to select optimal tuning parameters
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
# fit model
fit <- train(classe ~ ., data=training, method="rf", trControl=fitControl)
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.64%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    2    0    0    1 0.0007168459
## B   16 2826    6    0    0 0.0077247191
## C    0   16 2547    4    0 0.0077911959
## D    0    0   40 2369    3 0.0178275290
## E    0    0    1    5 2700 0.0022172949
```

```r
plot(fit$finalModel)
```

![](Preview-10b03c847806_files/figure-html/unnamed-chunk-10-1.png) 

```r
plot(fit)
```

![](Preview-10b03c847806_files/figure-html/unnamed-chunk-10-2.png) 

```r
#Randome Forest 3
#set.seed(1777)
#random_forest=randomForest(classe~.,data=training,ntree=500,importance=TRUE)
#random_forest
#plot(random_forest,main="Random Forest: Error Rate vs Number of Trees")

#model = train(classe~., method="rf", data=training)
#saveRDS(model, "rfmodel.RDS")
#model = readRDS("rfmodel.RDS")


#R Part Decision Tree
ModelDecT <- rpart(classe ~ ., data=training, method="class")
```

The forest and Trees model was selected.  The missclassification rate is less than 1%.  
The error rate of the model flattened at around 100 decision trees
The accuracy of the model was maximized with around 28 variables


####Step 9 - Assess the Models and Select the best model


```r
#K Nearest Neighbor
#fitKNN = predict(modelKNN, testing)
#mean(fitKNN != testing$classe)  ##Misclassification rate #8.4%
#confusionMatrix(testing$classe, fitKNN)

#Random Forest 1
#fitRF = predict(modelRF, testing)
#mean(fitRF != testing$classe)  ##Misclassification rate 
#confusionMatrix(testing$classe, fitRF)

#Random Forest 2 - with cross validation
fitRF2 = predict(fit, newdata=testing)
mean(fitRF2 != testing$classe)  ##Misclassification rate  #.007%
```

```
## [1] 0.00591354
```

```r
confusionMatrix(testing$classe, fitRF2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    3  946    0    0    0
##          C    0   13  841    1    0
##          D    0    0   12  792    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2851         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9864   0.9859   0.9987   1.0000
## Specificity            1.0000   0.9992   0.9965   0.9971   1.0000
## Pos Pred Value         1.0000   0.9968   0.9836   0.9851   1.0000
## Neg Pred Value         0.9991   0.9967   0.9970   0.9998   1.0000
## Prevalence             0.2851   0.1956   0.1739   0.1617   0.1837
## Detection Rate         0.2845   0.1929   0.1715   0.1615   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9989   0.9928   0.9912   0.9979   1.0000
```

```r
#Decision Tree
fitDecT1 = predict(ModelDecT, newdata=testing, type = "class")
mean(fitDecT1 != testing$classe)  ##Misclassification rate  #24.7%
```

```
## [1] 0.2606036
```

```r
confusionMatrix(testing$classe, fitDecT1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1235   55   44   41   20
##          B  157  568  125   64   35
##          C   16   73  690   50   26
##          D   50   80  118  508   48
##          E   20  102  116   38  625
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7394          
##                  95% CI : (0.7269, 0.7516)
##     No Information Rate : 0.3014          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6697          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8356   0.6469   0.6313   0.7247   0.8289
## Specificity            0.9533   0.9054   0.9567   0.9296   0.9335
## Pos Pred Value         0.8853   0.5985   0.8070   0.6318   0.6937
## Neg Pred Value         0.9307   0.9216   0.9005   0.9529   0.9678
## Prevalence             0.3014   0.1790   0.2229   0.1429   0.1538
## Detection Rate         0.2518   0.1158   0.1407   0.1036   0.1274
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8944   0.7761   0.7940   0.8271   0.8812
```



####Step 10 - Make Model Predictions

```r
# Make Predictions on the 20 test set records
preds <- predict(fit, newdata=testdata)

# convert predictions to character vector
preds <- as.character(preds)

# create function to write predictions to files
pml_write_files <- function(x) {
  n <- length(x)
  for(i in 1:n) {
    filename <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
  }
}

# create prediction files to submit
pml_write_files(preds)
preds
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

The model was 100% accurate on the 20 rows from the test data


####CONCLUSION

A 3-fold cross validation Forest and Trees model was selected as the final model.
The estimated out of sample missclassification error is only .007%% (1 - testing accuracy). 



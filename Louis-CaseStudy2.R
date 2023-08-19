###################### Case Study 2 #####################
setwd("C:/Users/lc4mo/OneDrive/Documents/Math 6333 Stat Learning")
load("Math_6333_Case_Study_2_Louis_Christopher.RData")
#1
train<-read.csv("C:/Users/lc4mo/OneDrive/Documents/Math 6333 Stat Learning/InsuranceData_train.csv",header = TRUE) 
valid <- read.csv("C:/Users/lc4mo/OneDrive/Documents/Math 6333 Stat Learning/InsuranceData_valid.csv",header = TRUE) 
#Check the dimensions of the data using the following. 
dim(train) #10,000 by 69 
dim(valid) #10,000 by 69 

#2
#install.packages("Information")
library(Information)
IV <- create_infotables(data=train, y="PURCHASE", ncore=2)
View(IV$Summary)
train_new <- train[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]
dim(train_new) #10,000 x 34
valid_new <- valid[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]
dim(valid_new) #10,000 x 34


#3
######## Variable clustering 
#install.packages("ClustOfVar")
#install.packages("reshape2")
#install.packages("plyr")
library(ClustOfVar)
library(reshape2)
library(plyr)
tree <- hclustvar(train_new [,!(names(train_new)=="PURCHASE")])
nvars <- 20
part_init<-cutreevar(tree,nvars)$cluster
kmeans<-
  kmeansvar(X.quanti=train_new[,!(names(train_new)=="PURCHASE")],init=part_init)
clusters <- cbind.data.frame(melt(kmeans$cluster), row.names(melt(kmeans$cluster)))
names(clusters) <- c("Cluster", "Variable")
clusters <- join(clusters, IV$Summary, by="Variable", type="left")
clusters <- clusters[order(clusters$Cluster),]
clusters$Rank <- ave(-clusters$IV, clusters$Cluster, FUN=rank)
View(clusters)
variables <- as.character(subset(clusters, Rank==1)$Variable)
#This will give you the final 20 variables that you will use for classification purposes.
variables

#4
#Create categorical variable NEWPurchase (our response) for train data
NEWPurchase = as.factor(ifelse(train_new$PURCHASE==1,"1","-1"))
train_new = data.frame(train_new, NEWPurchase)
#Remove PURCHASE from data set
names(train_new)
train_new = train_new[,-34]
str(train_new)
summary(train_new[,variables]) #Include in R markdown

#Create categorical variable NEWPurchase (our response) for valid data
NEWPurchase = as.factor(ifelse(valid_new$PURCHASE==1,"1","-1"))
valid_new = data.frame(valid_new, NEWPurchase)
#Remove PURCHASE from data set
names(valid_new)
valid_new = valid_new[,-34]
summary(valid_new[,variables]) #Include in R Markdown


#Visualizing our new train data with correlation analysis
CorrelationMatrix_train <- cor(train_new[,variables],
                               use="complete.obs")
#Rounding it to the second decimal place
round(CorrelationMatrix_train,2)
library(corrplot)
dev.off()
corrplot(CorrelationMatrix_train, tl.col= "black",
         title= "Correlation Plot Train", tl.cex=.5)


#Visualizing our new valid data with correlation analysis
CorrelationMatrix_valid <- cor(valid_new[,variables],
                               use="complete.obs")
#Rounding it to the second decimal place
round(CorrelationMatrix_valid,2)
library(corrplot)
dev.off()
corrplot(CorrelationMatrix_valid, tl.col= "black",
         title= "Correlation Plot Valid", tl.cex=.5)


#5
library(randomForest)
# Model 1
set.seed(123)
model1 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=1, ntree=10001)
model1  # OOB=20.16

# Model 2
set.seed(123)
model2 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=2, ntree=10001)
model2  # OOB=18.24

# Model 3
set.seed(123)
model3 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=3, ntree=10001)
model3  # OOB=18.18

# Model 4
set.seed(123)
RF_Model = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=4, ntree=10001)
RF_Model  # OOB=18.09

# Model 5
set.seed(123)
model5 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=5, ntree=10001)
model5  # OOB=18.16

# Model 6
set.seed(123)
model6 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=6, ntree=10001)
model6  # OOB=18.20

# Model 7
set.seed(123)
model7 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=7, ntree=10001)
model7  # OOB=18.10

# Model 8
set.seed(123)
model8 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=8, ntree=10001)
model8  # OOB=18.10

# Model 9
set.seed(123)
model9 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=9, ntree=10001)
model9  # OOB=18.14

# Model 10
set.seed(123)
model10 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=10, ntree=10001)
model10  # OOB=18.14

# Model 11
set.seed(123)
model11 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=11, ntree=10001)
model11  # OOB=18.23

# Model 12
set.seed(123)
model12 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=12, ntree=10001)
model12  # OOB=18.16

# Model 13
set.seed(123)
model13 = randomForest(NEWPurchase~.,
                      data=train_new[,c("NEWPurchase",variables)],
                      mtry=13, ntree=10001)
model13  # OOB=18.25

###### Model 4 has the lowest OOB (Best Model)
varImpPlot(model4)
library(caret)
sort(varImp(model4)[,1], decreasing = TRUE)

#Predicting with our same 20 variables?
yhat.rf = predict(RF_Model, newdata=valid_new[,c("NEWPurchase",variables)])
#yhat.rf
table(yhat.rf)

xtabrf = table(yhat.rf, valid_new$NEWPurchase)
library("e1071")
caret::confusionMatrix(xtabrf,mode="everything", positive = "1")

###ROC Curve with "pROC" library 
library(pROC)
plot(roc(valid_new$NEWPurchase, as.numeric(data.frame(yhat.rf)[,1])),
     print.auc=TRUE,col="blue",lwd=3,main="Random Forest ROC Curve for Insurance Test Data")

auc(roc(valid_new$NEWPurchase, as.numeric(data.frame(yhat.rf)[,1])),
    col="yellow", lwd=3, main="Random Forest ROC Curve for Insurance Test Data")
#AUC = 0.96


#6
# Support Vector Machine
library(e1071)
#First model
set.seed(123)
svm.model<-svm(NEWPurchase~.,data=train_new[,c("NEWPurchase",variables)],
               cost=0.01,kernel="polynomial",degree=3,probability=TRUE)
summary(svm.model)
#Predictions
yhat_svm1 = predict(svm.model,newdata=valid_new,probability = TRUE)
xtab_svm1 = table(yhat_svm1, valid_new$NEWPurchase)
library("e1071")
caret::confusionMatrix(xtab_svm1, positive = "1")

###ROC Curve with "pROC" library 
library(pROC)
plot(roc(valid_new$NEWPurchase, as.numeric(data.frame(yhat_svm1)[,1])),
     print.auc=TRUE,col="blue",lwd=3,main="SVM (Polynomial Kernel) ROC Curve for Insurance Test Data")


###### Second model
set.seed(123)
svm.model2<-svm(NEWPurchase~.,data=train_new[,c("NEWPurchase",variables)],
                cost=0.01,kernel="radial",gamma=0.000001,degree=3,probability=TRUE)
summary(svm.model2)
#Predictions
yhat_svm2 = predict(svm.model2,newdata=valid_new,probability = TRUE)
xtab_svm2 = table(yhat_svm2, valid_new$NEWPurchase)
library("e1071")
caret::confusionMatrix(xtab_svm2, positive = "1")

###ROC Curve with "pROC" library 
library(pROC)
plot(roc(valid_new$NEWPurchase, as.numeric(data.frame(yhat_svm2)[,1])),
     print.auc=TRUE,col="blue",lwd=3,main="SVM (Radial Kernel) ROC Curve for Insurance Test Data")


save.image("Math_6333_Case_Study_2_Louis_Christopher.RData")

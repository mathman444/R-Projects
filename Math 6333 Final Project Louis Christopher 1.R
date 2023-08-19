######################### Final Project ############################
################## Predicting Body Fat Percentage ##################
# Run setwd before running load.
setwd("C:/Users/lc4mo/OneDrive/Documents/Math 6333 Stat Learning")
load("Math_6333_Final_Project_Louis_Christopher.RData")
mydata = read.table("bodyfat.txt", header = TRUE)
head(mydata)
summary(mydata) #Can't have 0 body fat (need to correct)
dim(mydata) #250x16
str(mydata)

#Data Cleaning

#Convert any values <= 3 in column 2 (Pct.BF) into NA Values
mydata[,2][mydata[,2] <= 3] = NA
summary(mydata) #3 missing values

#Visualize the missing data
library(Amelia)
missmap(mydata)

#Remove missing values
mydata = na.omit(mydata)
dim(mydata) #247x16
missmap(mydata)

###Correlation Analysis
CorrelationMatrix <- cor(mydata,use="complete.obs")
#Rounding it to the second decimal place
round(CorrelationMatrix,2)
library(corrplot)
dev.off()
corrplot(CorrelationMatrix, tl.col= "black", title= "Correlation Plot", tl.cex=.5)

# Create Scatter Plots
plot(mydata$Density, mydata$Pct.BF, main="Scatterplot For Density",
     xlab="Density", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Density), col="red") # regression line (y~x)
lines(lowess(mydata$Density,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Age, mydata$Pct.BF, main="Scatterplot For Age",
     xlab="Age", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Age), col="red") # regression line (y~x)
lines(lowess(mydata$Age,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Weight, mydata$Pct.BF, main="Scatterplot For Weight",
     xlab="Weight", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Weight), col="red") # regression line (y~x)
lines(lowess(mydata$Weight,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Height, mydata$Pct.BF, main="Scatterplot For Height",
     xlab="Height", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Height), col="red") # regression line (y~x)
lines(lowess(mydata$Height,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Neck, mydata$Pct.BF, main="Scatterplot For Neck",
     xlab="Neck", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Neck), col="red") # regression line (y~x)
lines(lowess(mydata$Neck,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Chest, mydata$Pct.BF, main="Scatterplot For Chest",
     xlab="Chest", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Chest), col="red") # regression line (y~x)
lines(lowess(mydata$Chest,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Abdomen, mydata$Pct.BF, main="Scatterplot For Abdomen",
     xlab="Abdomen", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Abdomen), col="red") # regression line (y~x)
lines(lowess(mydata$Abdomen,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Waist, mydata$Pct.BF, main="Scatterplot For Waist",
     xlab="Waist", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Waist), col="red") # regression line (y~x)
lines(lowess(mydata$Waist,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Hip, mydata$Pct.BF, main="Scatterplot For Hip",
     xlab="Hip", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Hip), col="red") # regression line (y~x)
lines(lowess(mydata$Hip,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Thigh, mydata$Pct.BF, main="Scatterplot For Thigh",
     xlab="Thigh", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Thigh), col="red") # regression line (y~x)
lines(lowess(mydata$Thigh,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Knee, mydata$Pct.BF, main="Scatterplot For Knee",
     xlab="Knee", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Knee), col="red") # regression line (y~x)
lines(lowess(mydata$Knee,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Ankle, mydata$Pct.BF, main="Scatterplot For Ankle",
     xlab="Ankle", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Ankle), col="red") # regression line (y~x)
lines(lowess(mydata$Ankle,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Bicep, mydata$Pct.BF, main="Scatterplot For Bicep",
     xlab="Bicep", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Bicep), col="red") # regression line (y~x)
lines(lowess(mydata$Bicep,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Forearm, mydata$Pct.BF, main="Scatterplot For Forearm",
     xlab="Forearm", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Forearm), col="red") # regression line (y~x)
lines(lowess(mydata$Forearm,mydata$Pct.BF), col="blue") # lowess line (x,y)

plot(mydata$Wrist, mydata$Pct.BF, main="Scatterplot For Wrist",
     xlab="Wrist", ylab="Percentage Body Fat", pch=19)
# Add fit lines
abline(lm(mydata$Pct.BF~mydata$Wrist), col="red") # regression line (y~x)
lines(lowess(mydata$Wrist,mydata$Pct.BF), col="blue") # lowess line (x,y)

#Creating Histograms
dev.off()
par(mfrow=c(4,2))
hist(mydata$Density, main = "Distribution of Density",
     xlab = "Density", freq = FALSE)
lines(density(mydata$Density), col=1,lwd = 3)

hist(mydata$Pct.BF, main = "Distribution of Percentage Body Fat",
     xlab = "Pct.BF", freq = FALSE)
lines(density(mydata$Pct.BF), col=1,lwd = 3)

hist(mydata$Age, main = "Distribution of Age",
     xlab = "Age", freq = FALSE)
lines(density(mydata$Age), col=1,lwd = 3)

hist(mydata$Weight, main = "Distribution of Weight",
     xlab = "Weight", freq = FALSE)
lines(density(mydata$Weight), col=1,lwd = 3)

par(mfrow=c(4,2))
hist(mydata$Neck, main = "Distribution of Neck",
     xlab = "Neck", freq = FALSE)
lines(density(mydata$Neck), col=1,lwd = 3)

hist(mydata$Chest, main = "Distribution of Chest",
     xlab = "Chest", freq = FALSE)
lines(density(mydata$Chest), col=1,lwd = 3)

hist(mydata$Abdomen, main = "Distribution of Abdomen",
     xlab = "Abdomen", freq = FALSE)
lines(density(mydata$Abdomen), col=1,lwd = 3)

hist(mydata$Waist, main = "Distribution of Waist",
     xlab = "Waist", freq = FALSE)
lines(density(mydata$Waist), col=1,lwd = 3)

par(mfrow=c(4,2))
hist(mydata$Hip, main = "Distribution of Hip",
     xlab = "Hip", freq = FALSE)
lines(density(mydata$Hip), col=1,lwd = 3)

hist(mydata$Thigh, main = "Distribution of Thigh",
     xlab = "Thigh", freq = FALSE)
lines(density(mydata$Thigh), col=1,lwd = 3)

hist(mydata$Knee, main = "Distribution of Knee",
     xlab = "Knee", freq = FALSE)
lines(density(mydata$Knee), col=1,lwd = 3)

hist(mydata$Ankle, main = "Distribution of Ankle",
     xlab = "Ankle", freq = FALSE)
lines(density(mydata$Ankle), col=1,lwd = 3)

par(mfrow=c(4,2))
hist(mydata$Bicep, main = "Distribution of Bicep",
     xlab = "Bicep", freq = FALSE)
lines(density(mydata$Bicep), col=1,lwd = 3)

hist(mydata$Forearm, main = "Distribution of Forearm",
     xlab = "Forearm", freq = FALSE)
lines(density(mydata$Forearm), col=1,lwd = 3)

hist(mydata$Wrist, main = "Distribution of Wrist",
     xlab = "Wrist", freq = FALSE)
lines(density(mydata$Wrist), col=1,lwd = 3)

################# Train our Data #####################
#Create data partition.
library(caret)
set.seed(123)
indexdata <- createDataPartition(mydata$Pct.BF, p = .8,
                                    list = FALSE,times = 1)
head(indexdata)
trainingData<-mydata[indexdata,] #training the model
testingData<-mydata[-indexdata,] #Use for error metrics


##########Perform Multiple regression analysis #######

#tidyverse for easy data manipulation and visualization
library(tidyverse)
#leaps, for computing best subsets regression
library(leaps)
# Set up repeated k-fold cross-validation (Our K here is 3)
train.control <- trainControl(method = "cv", number = 3)

#####Train the model with forward selection
set.seed(123)
step.model1 <- caret::train(Pct.BF~., data = trainingData, 
                           method = "leapForward",
                           tuneGrid = data.frame(nvmax = 1:15),
                           trControl = train.control)

#Shows us each variable model with its corresponding error values
step.model1$results
step.model1$bestTune #Tells us the 5 variable model is best.

#Tells us which variables are included for each model.
summary(step.model1$finalModel)

#Gives us the coefficients for the 5 variable model.
coef(step.model1$finalModel,  5)
#Density Age Height Hip Thigh are included.

##For Forward Regression Analysis
set.seed(123)
Forwardmodel <- regsubsets(Pct.BF ~., data = trainingData,
                            nvmax = 15, method = "forward")
summary(Forwardmodel)

res.sumf <- summary(Forwardmodel)
res.sumf$adjr2
res.sumf$cp
res.sumf$bic

#Gives us the predictor number model in a data frame that has
#the highest adjusted R squared, lowest CP value, and 
#lowest BIC value.
data.frame(
  Adj.R2 = which.max(res.sumf$adjr2),
  CP = which.min(res.sumf$cp),
  BIC = which.min(res.sumf$bic))

#####Train the model using backward selection
set.seed(123)
step.model2 <- caret::train(Pct.BF~., data = trainingData, 
                           method = "leapBackward",
                           tuneGrid = data.frame(nvmax = 1:15),
                           trControl = train.control)

#Shows us each variable model with its corresponding error values
step.model2$results
step.model2$bestTune #Tells us the 4 variable model is best.

#Tells us which variables are included for each model.
summary(step.model2$finalModel)
summary(step.model2)

#Gives us the coefficients for the 1 variable model.
coef(step.model2$finalModel,  4)
# Density Height Hip Thigh

##For Backward Regression Analysis
set.seed(123)
Backwardmodel <- regsubsets(Pct.BF ~., data = trainingData,
                            nvmax = 15, method = "backward")
summary(Backwardmodel)

res.sumb <- summary(Backwardmodel)
res.sumb$adjr2
res.sumb$cp
res.sumb$bic

#Gives us the predictor number model in a data frame that has
#the highest adjusted R squared, lowest CP value, and 
#lowest BIC value.
data.frame(Adj.R2 = which.max(res.sumb$adjr2),
           CP = which.min(res.sumb$cp),
           BIC = which.min(res.sumb$bic))

######Train the model using Stepwise Subset Selection
set.seed(123)
step.model3 <- caret::train(Pct.BF~., data = trainingData, 
                                   method = "leapSeq",
                                   tuneGrid = data.frame(nvmax = 1:15),
                                   trControl = train.control)

#Shows us each variable model with its corresponding error values
step.model3$results
step.model3$bestTune #Tells us the 4 variable model is best.

#Tells us which variables are included for each model.
summary(step.model3$finalModel)

#Gives us the coefficients for the 4 variable model.
coef(step.model3$finalModel,  4)
# Density Height Hip Thigh

##For Stepwise Regression Analysis
set.seed(123)
Stepwisemodel <- regsubsets(Pct.BF ~., data = trainingData,
                             nvmax = 15, method = "seqrep")
summary(Stepwisemodel)

res.sum <- summary(Stepwisemodel)
res.sum$adjr2
res.sum$cp
res.sum$bic

#Gives us the predictor number model in a data frame that has
#the highest adjusted R squared, lowest CP value, and 
#lowest BIC value.
data.frame(
  Adj.R2 = which.max(res.sum$adjr2),
  CP = which.min(res.sum$cp),
  BIC = which.min(res.sum$bic)
)
#Chose model with 4 variables here.

mylrmodel = lm(Pct.BF~Density+Height+Hip+Thigh, data = trainingData)
mylrmodel
summary(mylrmodel)
library(car)
avPlots(mylrmodel)

##Prediction Interval (includes uncertainty)
predict(mylrmodel,testingData[-2], interval="predict")
yhat= predict(mylrmodel,testingData[-2], interval="predict")[,1]
yhat
length(yhat) #48
dim(testingData) #48 x 16

y=testingData$Pct.BF
length(y) #48

MSE = mean((y-yhat)^2)
MSE #8.04 We will use the linear model.

#Model Diagnostics
###Diagnostic Plots and Their Interpretations. 
par(mfrow = c(2, 2))
plot(mylrmodel)



#Trying a one variable model
set.seed(123)
lrmodeldensity <- lm(Pct.BF~Density, data = trainingData)
summary(lrmodeldensity)
##Prediction Interval (includes uncertainty)
predict(lrmodeldensity,testingData[-2], interval="predict")
yhat2= predict(lrmodeldensity,testingData[-2], interval="predict")[,1]
yhat2

y2=testingData$Pct.BF

MSEdensity = mean((y2-yhat2)^2)
MSEdensity #8.07


######################## K-mean Regression #######################
set.seed(123)
my_knn_model <- train(trainingData[-2], trainingData[,2], method = "knn",
                      preProcess=c("center", "scale"),
                      tuneGrid = expand.grid(k = c(3,5,7,9,11)))

my_knn_model
my_knn_model$bestTune #k = 7 is the best model

# Predict the labels of the test set
predictions<-predict(object=my_knn_model,testingData[,-2])

# Evaluate the predictions
#install.packages("Metrics")
library(Metrics) 
#MSE=(Sum_1^n{(y_i-hat(y_i))^2})/n
mse_test <-mean((testingData[,2] - predictions)^2)
mse_test #12.53 Highest MSE

###################### Lasso Regression #########################
#Removing the first column (the intercept)
x_train = model.matrix(Pct.BF~.,trainingData)[,-1]
head(x_train)
dim(x_train) #199x15

x_test = model.matrix(Pct.BF~.,testingData)[,-1]
head(x_test)
dim(x_test) #48x15

#Our response (What we want to predict)
y_train = trainingData$Pct.BF
y_train
length(y_train) #199

y_test = testingData$Pct.BF
y_test
length(y_test) #48

#Creating a grid of values to be used as our Lambda.
library(glmnet)
set.seed(123)
grid=10^seq(10,-2,length=100)

# The Lasso
set.seed(123)
lasso.mod=glmnet(x_train,y_train,alpha=1,lambda=grid)
lasso.mod
plot(lasso.mod,xvar="lambda",label=TRUE)
plot(lasso.mod,xvar="dev",label=TRUE)

#Using cross-validation to find the best lambda.
set.seed(123)
cv.lasso=cv.glmnet(x_train,y_train,alpha=1)
plot(cv.lasso)
bestlamlasso=cv.lasso$lambda.min
bestlamlasso
coef(cv.lasso) #The Best Model Coefficients
##Making Predictions
#Method 1 with best lambda chosen
lasso.pred=predict(lasso.mod, s = bestlamlasso, newx = x_test)
lasso.pred
dim(lasso.pred) #48x1

MSE_Lasso = mean((lasso.pred-y_test)^2)
MSE_Lasso #8.02


#Method 2 
lasso.pred2=predict(lasso.mod, newx = x_test)
lasso.pred2
dim(lasso.pred2)

RMSE2<-sqrt(apply((y_test-lasso.pred2)^2,2,mean))
plot(log(lasso.mod$lambda),RMSE2,type="b",xlab="Log(lambda)")
lam.best=lasso.mod$lambda[order(RMSE2)[1]]
lam.best
coef(lasso.mod,s=lam.best)

lasso2 = glmnet(x_train,y_train,alpha=1,lambda=grid)
lasso.coef=predict(lasso2,type="coefficients",s=lam.best)[1:16,]
lasso.coef
lasso.coef[lasso.coef!=0]

######################## Regression Tree ##########################
set.seed(123)
library(tree)
#Creating a regression tree with Pct.BF as the response variable.
tree.BF = tree(Pct.BF~., trainingData)
summary(tree.BF)

dev.off()
plot(tree.BF)
text(tree.BF,pretty=0)
#Density is the most important variable.

#Cross Validation for Regression Trees
cv.BF = cv.tree(tree.BF)
plot(cv.BF$size,cv.BF$dev,type='b')
prune.BF = prune.tree(tree.BF,best=4)
plot(prune.BF)
text(prune.BF,pretty=0)

yhat=predict(tree.BF,newdata = testingData)
plot(yhat,y)
abline(0,1)
#MSE value
mean((yhat-y)^2) #7.023


################### Bagging and Random Forests ####################

library(randomForest)
set.seed(123)
#mtry means we are considering all 15 predictor variables (bagging).
#Create a for-loop to find the best mtry value with lowest MSE.
bag.mse <- c(NA)
for (i in 1:15){
  bag.BF=randomForest(Pct.BF~.,data=trainingData,mtry=i,importance=TRUE)
  bag.mse[i]<-bag.BF$mse[500]}
bag.mse #mtry=15 has the lowest mse
#Create model with mtry=15
set.seed(123)
rf.model<-randomForest(Pct.BF~.,data=trainingData,mtry=15,importance=TRUE)
rf.model

##The MSR and % variance explained are based on OOB estimates.
yhat.bag = predict(rf.model, newdata = testingData)
plot(yhat.bag, y)
abline(0,1)
#We want the values to follow a linear line y=x.
#MSE
mean((yhat.bag-y)^2) #5.70

#Trying another model
set.seed(123)
bag.BF2=randomForest(Pct.BF~.,data=trainingData,
                        mtry=15,ntree=25)
yhat.bag = predict(bag.BF2, newdata = testingData)
#MSE
mean((yhat.bag-y)^2) #5.63

#Shows variables by importance.
#1st is Density, 2nd is Abdomen, 3rd is Waist.
importance(rf.BF)
##Variable Importance Plot
varImpPlot(rf.BF)

### Boosting (Usually performs better than bagging and random forests)
library(gbm)
set.seed(123)
boost.BF1=gbm(Pct.BF~.,data=trainingData,
                 distribution="gaussian",
                 n.trees=5000,interaction.depth=4)
summary(boost.BF1)
yhat.boost1=predict(boost.BF1,newdata=testingData,n.trees=5000)
#MSE
mean((yhat.boost1-y)^2) #5.03

dev.off()
#As Density increases, so Pct.BF decreases.
plot(boost.BF1,i="Density")
#As Height increases, the Pct.BF decreases.
plot(boost.BF1,i="Height")
#No correlation with abdomen and Pct.BF ?
plot(boost.BF1, i="Abdomen")


#Trying boost with interaction.depth = 3
set.seed(123)
boost.BF2=gbm(Pct.BF~.,data=trainingData,
              distribution="gaussian",
              n.trees=5000,interaction.depth=3)
summary(boost.BF2)
yhat.boost2=predict(boost.BF2,newdata=testingData,n.trees=5000)
#MSE
mean((yhat.boost2-y)^2) #5.12

#Trying boost with interaction.depth = 2
set.seed(123)
boost.BF3=gbm(Pct.BF~.,data=trainingData,
              distribution="gaussian",
              n.trees=5000,interaction.depth=2)
summary(boost.BF3)
yhat.boost3=predict(boost.BF3,newdata=testingData,n.trees=5000)
#MSE
mean((yhat.boost3-y)^2) #4.95

#Trying boost with interaction.depth = 1
set.seed(123)
boost.BF4=gbm(Pct.BF~.,data=trainingData,
              distribution="gaussian",
              n.trees=5000,interaction.depth=1)
summary(boost.BF4)
yhat.boost4=predict(boost.BF4,newdata=testingData,n.trees=5000)
#MSE
mean((yhat.boost4-y)^2) #6.54


#Another boosting but with shrinkage (Most accurate model)
library(gbm)
set.seed(123)
boost.BF5=gbm(Pct.BF~.,data=trainingData,
                 distribution="gaussian",n.trees=5000,
                 interaction.depth=2,shrinkage=0.015,
                 verbose=F)
yhat.boost5=predict(boost.BF5,newdata=testingData,n.trees=5000)
#MSE
mean((yhat.boost5-y)^2) #4.70 Lowest MSE 

#Shows variables by importance.
summary.gbm(boost.BF5)
boost.BF5

save.image("Math_6333_Final_Project_Louis_Christopher.RData")

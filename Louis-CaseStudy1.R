setwd("C:/Users/lc4mo/OneDrive/Documents/Math 6333 Stat Learning")
load("Math6333_Louis_CaseStudy1.RData")

#Part 1
##Reading the Data
mydata<-read.csv("CaseStudy1.csv",header = TRUE)

names(mydata)
head(mydata)
tail(mydata)
dim(mydata) ##142 x 14
class(mydata)
summary(mydata)
#We see 86 missing values for Post_TFI_Score

#Checking to see the structure of the data
#If there are any categorical variables
str(mydata)

#Changing Gender and Group to categorical variables.
mydata$Gender=as.factor(mydata$Gender)
contrasts(mydata$Gender) #0 male is reference, 1 female.
is.factor(mydata$Gender)
mydata$Group=as.factor(mydata$Group)
str(mydata)
#We see that Group has 3 levels so we need to change this to 2.

table(mydata$Group) #Getting a better view of whats going on.
levels(mydata$Group)

#Here we make the control category with a space equal to the
#control category without a space.
#levels(mydata$Group)[levels(TIDataRevisedAna$Low_Freq_Tinnitus)==""] <- "No"
levels(mydata$Group)[levels(mydata$Group)=="Control "] <- "Control"
levels(mydata$Group) #Checking to see if they're equal.

#Visualizing our missing values.
library(naniar)
library(ggplot2)
gg_miss_var(mydata, show_pct = TRUE)
gg_miss_var(mydata,facet = Group, show_pct = TRUE)


###Correlation Analysis
#We include all rows but exclude the 1st,2nd,
#and 10th columns
is.numeric(mydata$Post_TFI_Score)
CorrelationMatrix1 <- cor(mydata[,-c(1,2,10)],use="complete.obs")
#Pearson is the default.
#Rounding it to the second decimal place
round(CorrelationMatrix1,2)
library(corrplot)
dev.off()
corrplot(CorrelationMatrix1, tl.col= "black", title= "Correlation Plot", tl.cex=.5)
dev.off()
#Only include highest correlated variables
#pairs(mydata[,-c(1,2,10)])



#Part 2 (data imputation)

#This is how you would get the mean of a specific group.
#mean(mydata[mydata$Group=="Control",14], na.rm = TRUE)

#Getting the mean of the Post_TFI_Score Column.
mean(mydata[,14], na.rm = TRUE) #We get the mean without the NA values.

#Changing the missing values to the mean of the Post_TFI_Score column.
mydata$Post_TFI_Score[is.na(mydata$Post_TFI_Score)] <- mean(mydata[,14], na.rm = T)
#Checking to see if the missing values are filled in.
mydata$Post_TFI_Score

str(mydata$Post_TFI_Score)
#Showing values for just the Control Group.
mydata[mydata$Group=="Control",14]

#Making sure our Post_TFI_Score column is numeric.
mydata$Post_TFI_Score<-as.numeric(mydata$Post_TFI_Score)

#Creating our new column TFI_Reduction
mydata$TFI_Reduction <- mydata$Post_TFI_Score-mydata$Pre_TFI_Score
mydata$TFI_Reduction
names(mydata)

#Won't include Subject ID, Group, Pre TFI Score or 
#Post TFI Score in our regression analysis
newdata = mydata[,-c(1,2,13,14)]
names(newdata)

#Creating Histograms
par(mfrow=c(5,2))
hist(newdata$HHI_Score, main = "Distribution of HHI_Score",
     xlab = "HHI_Score", freq = FALSE)
lines(density(newdata$HHI_Score), col=1,lwd = 3)

hist(newdata$GAD, main = "Distribution of GAD",
     xlab = "GAD", freq = FALSE)
lines(density(newdata$GAD), col=1,lwd = 3)

hist(newdata$PHQ, main = "Distribution of PHQ",
     xlab = "PHQ", freq = FALSE)
lines(density(newdata$PHQ), col=1,lwd = 3)

hist(newdata$ISI, main = "Distribution of ISI",
     xlab = "ISI", freq = FALSE)
lines(density(newdata$ISI), col=1,lwd = 3)

hist(newdata$SWLS, main = "Distribution of SWLS",
     xlab = "SWLS", freq = FALSE)
lines(density(newdata$SWLS), col=1,lwd = 3)

hist(newdata$Hyperacusis, main = "Distribution of Hyperacusis",
     xlab = "Hyperacusis", freq = FALSE)
lines(density(newdata$Hyperacusis), col=1,lwd = 3)

hist(newdata$CFQ, main = "Distribution of CFQ",
     xlab = "CFQ", freq = FALSE)
lines(density(newdata$CFQ), col=1,lwd = 3)

hist(newdata$Age, main = "Distribution of Age",
     xlab = "Age", freq = FALSE)
lines(density(newdata$Age), col=1,lwd = 3)

hist(newdata$Duration, main = "Distribution of Duration of Tinnitus",
     xlab = "Duration of Tinnitus", freq = FALSE)
lines(density(newdata$Duration), col=1,lwd = 3)



#Visualizing our new data with correlation analysis
CorrelationMatrix2 <- cor(newdata[,-c(8)],use="complete.obs")
#Rounding it to the second decimal place
round(CorrelationMatrix2,2)
library(corrplot)
dev.off()
corrplot(CorrelationMatrix2, tl.col= "black", title= "Correlation Plot", tl.cex=.5)


#Part 4
#Create data partition.
library(caret)
set.seed(123)
trainnewdata <- createDataPartition(newdata$TFI_Reduction, p = .8,
                                    list = FALSE,times = 1)
head(trainnewdata)
trainingData<-newdata[trainnewdata,] #training the model
testingData<-newdata[-trainnewdata,] #Use for error metrics



#Part 5
#Perform Multiple regression analysis

#tidyverse for easy data manipulation and visualization
library(tidyverse)
#leaps, for computing best subsets regression
library(leaps)

#"leapBackward", to fit linear regression with backward selection
#"leapForward", to fit linear regression with forward selection
#"leapSeq", to fit linear regression with stepwise selection .

# Set up repeated k-fold cross-validation (Our K here is 10)
train.control <- trainControl(method = "cv", number = 10)

# Train the model using Backward Selection
step.modelbackward <- caret::train(TFI_Reduction~., data = trainingData,
                           method = "leapBackward",
                           tuneGrid = data.frame(nvmax = 1:10),
                           trControl = train.control
)
#Shows us each variable model with its corresponding error values
step.modelbackward$results

#Tells us that the 3 variable model is best since the errors are
#lower and the R squared is highest
step.modelbackward$bestTune

#Tells us which variables are included for each model.
summary(step.modelbackward$finalModel)

#Gives us the coefficients for the 3 variable model.
coef(step.modelbackward$finalModel,  3)

#For Backward Regression Analysis
backwardmodels <- regsubsets(TFI_Reduction ~., data = trainingData, nvmax = 10, method = "backward")
summary(backwardmodels)


res.sum1 <- summary(backwardmodels)

res.sum1$adjr2
##After Model with 6 variables R2 adjusted
#does not increase significantly.
res.sum1$cp
res.sum1$bic
#Gives us the predictor number model in a data frame that has
#the highest adjusted R squared, lowest CP value, and 
#lowest BIC value.
data.frame(
  Adj.R2 = which.max(res.sum1$adjr2),
  CP = which.min(res.sum1$cp),
  BIC = which.min(res.sum1$bic)
)
#Choose the 3 variable model here.

## Train the model using Forward Selection
step.modelforwards <- caret::train(TFI_Reduction~., data = trainingData,
                                   method = "leapForward",
                                   tuneGrid = data.frame(nvmax = 1:10),
                                   trControl = train.control
)
#Shows us each variable model with its corresponding error values
step.modelforwards$results

#Tells us that the 3 variable model is best since the errors are
#lower and the R squared is highest
step.modelforwards$bestTune

#Tells us which variables are included for each model.
summary(step.modelforwards$finalModel)

#Gives us the coefficients for the 3 variable model.
coef(step.modelforwards$finalModel, 3)

##For Forward Regression Analysis
forwardmodels <- regsubsets(TFI_Reduction ~., data = trainingData, nvmax = 10, method = "forward")
summary(forwardmodels)

res.sum2 <- summary(forwardmodels)

res.sum2$adjr2
##After Model with 6 variables R2 adjusted
#does not increase significantly.
res.sum2$cp
res.sum2$bic
#Gives us the predictor number model in a data frame that has
#the highest adjusted R squared, lowest CP value, and 
#lowest BIC value.
data.frame(
  Adj.R2 = which.max(res.sum2$adjr2),
  CP = which.min(res.sum2$cp),
  BIC = which.min(res.sum2$bic)
)
#Choose the 3 variable model here.

##Train the model using Stepwise Subset Selection
step.modelstepwise <- caret::train(TFI_Reduction~., data = trainingData, 
                                   method = "leapSeq",
                                   tuneGrid = data.frame(nvmax = 1:10),
                                   trControl = train.control)

#Shows us each variable model with its corresponding error values
step.modelstepwise$results

#Tells us that the 8 variable model is best since the errors are
#lower and the R squared is highest
step.modelstepwise$bestTune

#Tells us which variables are included for each model.
summary(step.modelstepwise$finalModel)

#Gives us the coefficients for the 8 variable model.
coef(step.modelstepwise$finalModel,  8)

##For Stepwise Regression Analysis
Stepwisemodels <- regsubsets(TFI_Reduction ~., data = trainingData,
                             nvmax = 10, method = "seqrep")
summary(Stepwisemodels)


res.sum3 <- summary(Stepwisemodels)

res.sum3$adjr2
##After Model with 5 variables R2 adjusted
#does not increase significantly.
res.sum3$cp
res.sum3$bic

#Gives us the predictor number model in a data frame that has
#the highest adjusted R squared, lowest CP value, and 
#lowest BIC value.
data.frame(
  Adj.R2 = which.max(res.sum3$adjr2),
  CP = which.min(res.sum3$cp),
  BIC = which.min(res.sum3$bic)
)
#Chose model with 3 variables here.
#Gives us the coefficients for the 3 variable model.
coef(step.modelstepwise$finalModel,  3)


#Our model will include the predictors
#GAD, ISI, and SWLS
mymodel = lm(TFI_Reduction~GAD+ISI+SWLS,data=trainingData)
mymodel
summary(mymodel)
library(car)
avPlots(mymodel)


##Part 6
#Model Diagnostics
###Diagnostic Plots and Their Interpretations. 
par(mfrow = c(2, 2))
plot(mymodel)

#The four diagnostic plots show residuals in four different ways:

#Plot 1. Residuals vs Fitted. Used to check the linear relationship assumptions. 
#A horizontal line, without distinct patterns is an indication for a linear relationship, what is good.

#Plot 2. Normal Q-Q. Used to examine whether the residuals are normally distributed. 
#It's good if residuals points follow the straight dashed line.

#Plot 3. Scale-Location (or Spread-Location). Used to check the homogeneity of variance of the residuals (homoscedasticity).
#Horizontal line with equally spread points is a good indication of homoscedasticity. 

#Plot 4. Residuals vs Leverage. Used to identify influential cases, that is extreme values that might influence the 
#regression results when included or excluded from the analysis. Presence of influential values can be in two main types.
#Outliers: extreme values in the outcome (y) variable
#High-leverage points: extreme values in the predictors (x) variable


#install.packages("broom")
library(broom)
library(dplyr)
##From the broom package
model.diag.metrics <- augment(mymodel)
model.diag.metrics %>% top_n(5, wt = .cooksd)
# Standardized Residuals vs Leverage
#The points 37, 57, and 62 stand out.
plot(mymodel, 5)

##When data points have high Cook's distance scores and are to the upper 
#or lower right of the leverage plot, 
#they have leverage meaning they are influential to the regression results. 
#The regression results will be altered if we exclude those cases. 

##################################################################
#Influential Points
###################################################################
#An influential value is a value, which inclusion or exclusion can alter the results of the regression analysis. 
#Such a value is associated with a large residual.

#Not all outliers (or extreme data points) are influential in linear regression analysis.

#Statisticians have developed a metric called Cook's distance to determine the influence of a value. 
#This metric defines influence as a combination of leverage and residual size.

#A rule of thumb is that an observation has high influence if 
#Cook's distance exceeds 4/(n - p - 1)(P. Bruce and Bruce 2017), 
#where n is the number of observations and p the number of predictor variables.
dim(trainingData)

# Cook's distance plotted
#We see that the points 37, 57, and 62 have a
#high Cook's distance.
dev.off()
plot(mymodel, 4)


#install.packages("broom")
library(broom)
model.diag.metrics <- augment(mymodel)
#Ex: we can check for Cases with elevated Cook's D and std.residual
#Here we checked rows 25-32?
model.diag.metrics[25:32,]



##Part 7
#GAD,ISI,SWLS interpret the coefficients



##Part 8
###Confidence and Prediction Interval 
predict(mymodel,testingData, interval="confidence")

##Prediction Interval (includes uncertainty)
testingData
predict(mymodel,testingData[1:10], interval="predict")
yhat= predict(mymodel,testingData[1:10], interval="predict")[,1]
yhat
length(yhat) #28
dim(testingData) #28 x 11

y=testingData$TFI_Reduction
length(y)

MSE = mean((y-yhat)^2)
MSE
sqrt(MSE) #17.04656



#Part 9
##K-mean Regression
set.seed(123)
my_knn_model <- train(trainingData[1:10], trainingData[,11], method = "knn",
                      preProcess=c("center", "scale"),
                      tuneGrid = expand.grid(k = c(2, 4, 6, 8, 10)))

my_knn_model
my_knn_model$bestTune #k = 10 is the best model

# Predict the labels of the test set
predictions<-predict(object=my_knn_model,testingData[,1:10])

# Evaluate the predictions
#install.packages("Metrics")
library(Metrics) 
# Calculating RMSE using rmse()          
result = rmse(testingData[,11],predictions) 
result #16.05517
#*MSE=(Sum_1^n{(y_i-hat(y_i))^2})/n
mse_test <-mean((testingData[,11] - predictions)^2)
mse_test #257.7685

#*MAE=(Sum_1^n{|y_i-hat(y_i)|})/n
mae_test <- caret::MAE(trainingData[,11], predictions)
mae_test #17.65808

##*RMSE=sqrt(MSE)
rmse_test<- caret::RMSE(testingData[,11], predictions)
rmse_test #16.05517

cat("Test MSE: ", mse_test, "Test MAE: ", mae_test, "Test RMSE: ", rmse_test)

#*Plotting Test Predictions

XX<- 1:length(testingData[,11])
XX

plot(XX, testingData[,11], col = "red", type = "l", lwd=2,
     main = "TFI Reduction test data prediction")

lines(XX, predictions, col = "blue", lwd=2)

#?????
legend("topright",  legend = c("", ""), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))

grid()


#Save Workspace
save.image("Math6333_Louis_CaseStudy1.RData")

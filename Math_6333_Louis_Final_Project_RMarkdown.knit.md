---
title: "Predicting Body Fat Percentage"
author: "Louis Christopher"
date: "2022-11-23"
output: html_document
---



## Introduction

The percent of a man's body that is fat is a matter of concern for health, fitness, and longevity. Elevated body fat percentages have been shown to impair the body's responsiveness to insulin, increase the risk of heart attacks, strokes, high blood pressure, cancer, diabetes, osteoarthritis, fatty liver, and depression. Therefore, it is important for a man to know his body fat percentage. The problem is that it is difficult, expensive, and time consuming to get an accurate reading. In this study we used Brigham Young University's Human Performance Research Center's data set which contains 250 male observations and 16 measurements. They accurately measured body fat percentage (denoted Pct.BF, ranging from 0.0-47.5%) along with Density (ranging from 0.9950-1.1089g/ml), Age (ranging from 22-81years), Weight (ranging from 118.5-262.8lb), Height (ranging from 64.00-77.75in), Neck (ranging from 31.10-43.90cm), Chest (ranging from 79.30-128.30cm), Abdomen (ranging from 69.40-126.20cm), Waist (27.32-49.69cm), Hip (ranging from 85.00-125.60cm), Thigh (ranging from 47.20-74.40cm), Knee (ranging from 33.00-46.00cm), Ankle (ranging from 19.10-33.90cm), Bicep (ranging from 24.80-39.10cm), Forearm (ranging from 21.00-34.9cm), and Wrist (ranging from 15.80-21.40cm). Our goal is to create a model that accurately predicts body fat percentage in men based on the other various measures. We also wish to know which of these measures contributes the most to our model (are most correlated to predicting body fat percentage).

## Literature Review

According to Harvard's T.H. Chan School of Public Health (2016), the most common and easy approach to determine whether someone is fat, or fit, is the body mass index (BMI). This is calculated from a ratio of weight to height (weight(lb)/height(in\^2) multiplied by 703). But this method is indirect, so it doesn't distinguish between body fat and lean body mass (muscle). Another method is measuring waist circumference. It is easy, inexpensive, and strongly correlated with body fat. Some limitations are that the procedure has not been standardized and is hard and less accurate in individuals with a BMI of 35 or higher.

Waist to hip ratio, and skinfold thickness are other relatively easy methods to predict body fat. But these also lack accuracy, are prone to measurement error, and are hard to measure with persons having a BMI of 35 or higher. Bio-electric Impedance (BIA) is safe and inexpensive but it's hard to calibrate and not as accurate as other methods. Underwater Weighing (Densitometry) is accurate but is time consuming as it requires individuals to be submerged in water. Air-Displacement Plethysmography uses a similar method as underwater weighing but in air. This method is accurate but expensive. The Dilution Method (Hydrometry) is low cost, accurate, and safe, but "the ratio of body water to fat-free mass may change during illness, dehydration, or weight loss, decreasing accuracy". More sophisticated methods like magnetic resonance imaging (MRI), and dual energy X-ray absorptiometry are accurate but are usually only performed in research settings and pose problems to individuals who cannot tolerate radiation.

Merrill Z (2020), and colleagues used backwards step-wise regression analysis on a data set of 228 adults between the ages of 21 and 70 with BMI between 18.5 and 40.0kg/m\^2 with predictors age, BMI, and "several anthropometric and skinfold measurements". They found that the final statistical regression model to predict body fat in males included age, BMI, hip circumference, mid-thigh circumference, hand thickness, subscapular skinfold, vertical abdominal skinfold, and thigh skinfold. They noted that, "the models predicted body fat percentage in the testing group with average errors of less than 0.10% body fat in males and females, while the four previously existing methods (Durnin, Hodgdon, Jackson, and Woolcott) significantly underestimated or overestimated body fat in both genders, with errors ranging between 2% and 10%".

Fan Z (2022) and colleagues researched which statistical learning methods were best for body fat prediction accuracy. They used four well-known machine learning models, Multi-Layer Perceptron (denoted MLP, a type of Artificial Neural Network), Support Vector Machine (SVM), Random Forests (RF) and eXtreme Gradient Boosting (XGBoost) combined with three feature extraction approaches, Factor Analysis (FA), Principal Components Analysis (PCA), and Independent Component Analysis (ICA) to predict body fat percentage. There were two cases. Case 1 was based on anthropometric measurements, while Case 2 was based on physical examination and laboratory measurements.

They found that, "in case 1 XGBoost with FA had the best approximation ability and high efficiency" and in case 2 they noted that "PCA was the most effective in improving model performance". Fan and colleagues noted that, "although the MLP with PCA had the best prediction accuracy, it required significantly more computation time. This means XGBoost is more appropriate for real-world applications, given its similar prediction accuracy and greater efficiency. Statistical analysis based on the Wilcoxon rank-sum test confirmed that feature extraction significantly improved the performance of MLP, SVM and XGBoost". The researchers also noted that results obtained by XGBoost with PCA feature extraction "could be used as the baseline for future research in related areas" and concluded that "methods of improving the prediction model (e.g. an improved MLP), using XGBoost with PCA as a baseline for body fat prediction, also need to be investigated".

Stevens J (2016) and colleagues used data from the 1999--2006 National Health and Nutrition Examination Survey (NHANES) to predict body fat percentage. The researchers sample included 21,099 men and women 8 years and older after exclusions. They used Dual-emission X-ray absorptiometry (DXA) assessed body fat percentage as the response variable to develop 14 equations for each gender that included between 2 and 10 anthropometrics (predictors). The predictors were age, height, weight, triceps and subscapular skinfolds, waist, maximal calf, arm and thigh circumferences, and upper arm and upper leg lengths. Also noting that, "age was used as a continuous variable and as a dichotomous variable indicating youth (8--19) or adult (≥20 years)".

Their models were developed using the Least Absolute Shrinkage and Selection Operator (LASSO) and validated in a 25% withheld sample randomly selected from 11,884 males or 9,215 females. In the final models, the researchers saw that $R^2$ ranged from 0.664 to 0.845 in males and 0.748 to 0.809 in females. Stevens and colleagues found that, "$R^2$ was not notably improved by development of equations within, rather than across, age and ethnic groups. Systematic over or under estimation of percent body fat by age and ethnic groups was within 1 percentage point". It is also worth noting that, "seven of the fourteen gender-specific models had R2 values above 0.80 in males and 0.795 in females and exhibited low bias by age, race/ethnicity and body mass index (BMI)". They also discovered that BMI alone produced an $R^2$ of 0.430 in males and 0.656 in females. At the end of the study they mentioned that when adding the predictors "triceps and subscapular skinfolds to the candidate variables of demographics, height, weight, and BMI improved performance more than the addition of up to four circumference measurements".

## Methods

The data set of 250 male observations was split into two sets, a training set (80% of the original data set) and a validation set (the other 20%). Our response variable is percentage body fat (denoted Pct.BF) and all other variables are considered as predictors. The data set was then analyzed using descriptive statistics. Missingness plots were used to view any missing data and scatter plots were used to identify linearity or non-linearity in the data. Correlation plots were also used to determine any correlation between the predictors and the response (Pct.BF).

The data analysis methods used to predict body fat percentage in this study were Linear Regression, KNN Regression, Least Absolute Shrinkage and Selection Operator (LASSO), and Regression Trees which included Bagged Random Forests and Boosted Regression Trees. After creating these models we compared their performance based on their Mean Squared Error ($MSE$) values. The $MSE$ of an estimator measures the average squared difference between the estimated values and the actual values.

For linear regression we used forward, backward, and stepwise selection to train and find the optimal model. This optimal model was chosen based on BIC, AIC (CP), and adjusted $R^2$ analysis. Linear regression assumes that our predictors are linearly related to our response (BF.Pct). If this assumption is violated then our model must be non-linear and one of the other methods may provide a better model. The LASSO was used with cross validation to select the smallest lambda value that minimized the $l_1$ norm. By using the LASSO some of the coefficients $\beta_j$ values may end up being zero. This can produce a model that has high predictive power and is simple to interpret. The LASSO also assumes a linear relationship between the predictors and response.

For KNN regression we trained the model with values of k = 3,5,7,9,11, pre-processed the data by centering and scaling it, and then selected the model with the lowest $MSE$. KNN works by approximating the association between independent variables and the continuous response by averaging the observations in the same neighborhood. KNN regression assumes that observations which exist in close proximity to each other are highly similar. If observations are far apart, our data will suffer from the 'curse of dimensionality' and our model will have low predictive power. The KNN is a black box model, so we won't know the exact relationship between the predictors and response (i.e linear or nonlinear).

To create a regression tree we used cross validation and pruned it using a best value of 4. Regression trees work by dividing the predictor space into $J$ distinct non-overlapping regions $R_1,R_2,...,R_j$, and then for every observation that falls into the region $R_j$, we make the same prediction which is the mean of the response values for the training observations in $R_j$. We also performed Bagged Random Forest Regression with mtry values ranging from 1 to 15 and selected the model with the lowest out of bag estimate ($OOB$). Boosted Regression Trees were also created using interaction depth values ranging from 1 to 4, a Gaussian distribution with 5000 trees, and shrinkage penalties varying from 0.01 to 0.05. We selected the best boosted regression tree based on having the lowest $MSE$.

Bagging involves creating multiple copies of the original training data set using the bootstrap, fitting a separate decision tree to each copy, and then combining all of the trees in order to create a single predictive model. Boosting works in a similar way, except that the trees are grown sequentially. This means that each tree is grown using information from previously grown trees. Random Forests build a number of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, a random selection of m predictors is chosen as split candidates from the full set of $p$ predictors. The split is allowed to use only one of those $m$ predictors. A fresh selection of $m$ predictors is taken at each split, and typically we choose $m\approx\sqrt{p}$. No formal distributional assumptions are made for Regression Trees or Random Forests. These can handle linear and nonlinear data well. It should be noted that Random Forests and Boosted Regression Trees are also considered black box models.

## Results

In our initial analysis of our data we see below in Figure (1) that our response variable percentage body fat (Pct.BF) has a minimum value of 0% and a maximum value of 47.50% with a mean of 19.03% and a median of 19.20%. Density has a minimum value of 0.995(g/ml) and a maximum value of 1.109(g/ml) with a mean of 1.056(g/ml) and median of 1.055(g/ml). Age has a minimum value of 22 years and a maximum of 81 years with a mean age of 44.88 years and a median of 43 years. Weight has a minimum value of 118.5 pounds and a maximum of 262.8 pounds with a mean weight of 178.1 pounds and median weight of 176.1 pounds. Similarly, summary statistics for the other predictors are listed below.


```
##     Density          Pct.BF           Age            Weight     
##  Min.   :0.995   Min.   : 0.00   Min.   :22.00   Min.   :118.5  
##  1st Qu.:1.042   1st Qu.:12.43   1st Qu.:35.25   1st Qu.:158.5  
##  Median :1.055   Median :19.20   Median :43.00   Median :176.1  
##  Mean   :1.056   Mean   :19.03   Mean   :44.88   Mean   :178.1  
##  3rd Qu.:1.070   3rd Qu.:25.20   3rd Qu.:54.00   3rd Qu.:196.8  
##  Max.   :1.109   Max.   :47.50   Max.   :81.00   Max.   :262.8  
##      Height           Neck           Chest           Abdomen      
##  Min.   :64.00   Min.   :31.10   Min.   : 79.30   Min.   : 69.40  
##  1st Qu.:68.25   1st Qu.:36.40   1st Qu.: 94.25   1st Qu.: 84.53  
##  Median :70.00   Median :38.00   Median : 99.60   Median : 90.90  
##  Mean   :70.30   Mean   :37.94   Mean   :100.66   Mean   : 92.29  
##  3rd Qu.:72.25   3rd Qu.:39.40   3rd Qu.:105.30   3rd Qu.: 99.17  
##  Max.   :77.75   Max.   :43.90   Max.   :128.30   Max.   :126.20  
##      Waist            Hip             Thigh            Knee      
##  Min.   :27.32   Min.   : 85.00   Min.   :47.20   Min.   :33.00  
##  1st Qu.:33.28   1st Qu.: 95.50   1st Qu.:56.00   1st Qu.:36.92  
##  Median :35.79   Median : 99.30   Median :58.95   Median :38.45  
##  Mean   :36.33   Mean   : 99.65   Mean   :59.25   Mean   :38.53  
##  3rd Qu.:39.05   3rd Qu.:103.17   3rd Qu.:62.25   3rd Qu.:39.88  
##  Max.   :49.69   Max.   :125.60   Max.   :74.40   Max.   :46.00  
##      Ankle           Bicep          Forearm          Wrist      
##  Min.   :19.10   Min.   :24.80   Min.   :21.00   Min.   :15.80  
##  1st Qu.:22.00   1st Qu.:30.20   1st Qu.:27.30   1st Qu.:17.60  
##  Median :22.80   Median :32.00   Median :28.70   Median :18.30  
##  Mean   :23.07   Mean   :32.22   Mean   :28.66   Mean   :18.22  
##  3rd Qu.:24.00   3rd Qu.:34.30   3rd Qu.:30.00   3rd Qu.:18.80  
##  Max.   :33.90   Max.   :39.10   Max.   :34.90   Max.   :21.40
```

Figure (1): Summary statistics for our data are listed above. The statistics include minimum and maximum values, along with the 1st and 3rd quadrant, and the median and mean.

Since we saw that percentage body fat had a minimum value of 0, which is impossible, we treated this value and any value less than or equal to 3% as a NA (not available) value. This is because any value less than 3% would be considered very abnormal. In Figure (2) below we generated summary statistics once again. We now see that Pct.BF has a minimum value of 3.70% and a maximum value of 47.50% along with a median of 19.20% and a mean of 19.25%. We also see beneath Pct.BF's maximum value that there are 3 NA values. The other predictors summary statistics are also listed below.


```
##     Density          Pct.BF           Age            Weight     
##  Min.   :0.995   Min.   : 3.70   Min.   :22.00   Min.   :118.5  
##  1st Qu.:1.042   1st Qu.:12.70   1st Qu.:35.25   1st Qu.:158.5  
##  Median :1.055   Median :19.20   Median :43.00   Median :176.1  
##  Mean   :1.056   Mean   :19.25   Mean   :44.88   Mean   :178.1  
##  3rd Qu.:1.070   3rd Qu.:25.25   3rd Qu.:54.00   3rd Qu.:196.8  
##  Max.   :1.109   Max.   :47.50   Max.   :81.00   Max.   :262.8  
##                  NA's   :3                                      
##      Height           Neck           Chest           Abdomen      
##  Min.   :64.00   Min.   :31.10   Min.   : 79.30   Min.   : 69.40  
##  1st Qu.:68.25   1st Qu.:36.40   1st Qu.: 94.25   1st Qu.: 84.53  
##  Median :70.00   Median :38.00   Median : 99.60   Median : 90.90  
##  Mean   :70.30   Mean   :37.94   Mean   :100.66   Mean   : 92.29  
##  3rd Qu.:72.25   3rd Qu.:39.40   3rd Qu.:105.30   3rd Qu.: 99.17  
##  Max.   :77.75   Max.   :43.90   Max.   :128.30   Max.   :126.20  
##                                                                   
##      Waist            Hip             Thigh            Knee      
##  Min.   :27.32   Min.   : 85.00   Min.   :47.20   Min.   :33.00  
##  1st Qu.:33.28   1st Qu.: 95.50   1st Qu.:56.00   1st Qu.:36.92  
##  Median :35.79   Median : 99.30   Median :58.95   Median :38.45  
##  Mean   :36.33   Mean   : 99.65   Mean   :59.25   Mean   :38.53  
##  3rd Qu.:39.05   3rd Qu.:103.17   3rd Qu.:62.25   3rd Qu.:39.88  
##  Max.   :49.69   Max.   :125.60   Max.   :74.40   Max.   :46.00  
##                                                                  
##      Ankle           Bicep          Forearm          Wrist      
##  Min.   :19.10   Min.   :24.80   Min.   :21.00   Min.   :15.80  
##  1st Qu.:22.00   1st Qu.:30.20   1st Qu.:27.30   1st Qu.:17.60  
##  Median :22.80   Median :32.00   Median :28.70   Median :18.30  
##  Mean   :23.07   Mean   :32.22   Mean   :28.66   Mean   :18.22  
##  3rd Qu.:24.00   3rd Qu.:34.30   3rd Qu.:30.00   3rd Qu.:18.80  
##  Max.   :33.90   Max.   :39.10   Max.   :34.90   Max.   :21.40  
## 
```

Figure (2): Above we see the summary statistics for our data with values of Pct.BF less than or equal to 3% as a NA value. We can see there are 3 observations meeting this criteria as listed underneath the maximum value of the Pct.BF column.

Looking at our data set visually with the missing data we generated a missingness map. As we can see in Figure (3) below, our 3 missing values account for roughly 0% of all of our observed data.

<img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-3-1.png" width="672" />

Figure (3): Missingness map above shows us that there are some missing values in our data set but they account for roughly 0% of our data.

Since our 3 missing values account for roughly 0% of all of our observed data, we decided to remove the observations associated with the 3 NA values. Thus, removing the 3 observations will leave us with 247 observations instead of the original 250. Generated below in Figure (4) is a new missingness map. As we can see there are no more missing values.

<img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-4-1.png" width="672" />

Figure (4): Missingness map above shows there are no more missing values for our data set.

After cleaning our data set we wanted to see the correlation between the predictors and response (Pct.BF). Below in Figure (5) is a correlation plot. Cool colors like dark blue/green indicate a high positive correlation, and warm colors like dark red/orange indicate a high negative correlation. In our correlation plot we see that Density has a highly negative correlation with Pct.BF. We also see that Abdomen, Waist, Chest, Weight, and Hip have a highly positive correlation with Pct.BF.

<img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-5-1.png" width="672" />

Figure (5): Seen above the correlation plot for our data. Cool colors like dark blue/green indicate a high positive correlation, and warm colors like dark red/orange indicate a high negative correlation.

Next we investigated the linearity of each predictor to the response. We did this by generating scatter plots with a red linear regression line and a green LOWESS line. The linear regression line indicates a linear relationship between the independent variable (one of our predictors) on our x-axis to our dependent variable (our response Pct.BF) on our y-axis. The LOWESS (Locally Weighted Scatterplot Smoothing) line is a method of regression analysis which fits a smooth line through a scatter plot. The green LOWESS line can indicate a nonlinear relationship which is useful to compare against our red linear regression line. If the green LOWESS line and red linear regression line look very similar, then we expect the predictor to have a linear relationship with Pct.BF.

Below in Figure (6) we see that Density and Pct.BF have a strong negative linear relationship. Both the LOWESS and linear regression line look almost identical. Age and Pct.BF look to have a nonlinear relationship. Weight and Pct.BF appear to have a slight positive nonlinear relationship. Height and Pct.BF have a slight negative nonlinear relationship. Neck and Pct.BF have a slight positive nonlinear relationship. Chest and Pct.BF look like they have a positive linear relationship. This is also seen in the scatter plots regarding Abdomen and Waist. Hip and Pct.BF appear have a slight positive nonlinear relationship.

The scatter plots for Thigh, Knee, Forearm, Bicep, and Wrist look to have a slight positive nonlinear relationship with Pct.BF. Ankle and Pct.BF appear to have a slight positive nonlinear relationship through most of the data points but towards the end the LOWESS turns into a negative linear relationship because of a couple outlier points. But, like a lot of the other scatter plots its hard to discern a true pattern in the data when it looks fairly random.

<img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-6-1.png" width="672" /><img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-6-2.png" width="672" /><img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-6-3.png" width="672" /><img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-6-4.png" width="672" />

Figure (6): Above we generated scatter plots fitted with a red linear regression line and a green LOWESS line. The linear regression line indicates a linear relationship between the independent variable (one of our predictors) on our x-axis to our dependent variable (our response Pct.BF) on our y-axis. The LOWESS (Locally Weighted Scatterplot Smoothing) line is a method of regression analysis which fits a smooth line through a scatter plot. The green LOWESS line can indicate a nonlinear relationship which is useful to compare against our red linear regression line.


When performing multiple linear regression we're trying to find a multiple linear regression model of the form $E[y|x]=B_0+B_1x+B_2x+...+B_nx+\epsilon$ where our x variables (predictors) are linearly related to our response variable y. The $\epsilon$ is our error term (noise in our model) which is normally distributed with a mean of zero and a variance of $\sigma^2$.

Performing the forward selection method resulted in a model with 5 predictors which were Density, Age, Height, Hip, and Thigh. Performing forward selection analysis we saw that the 6 variable model had the highest adjusted $R^2$ at 0.9979913, the 5 variable model had the lowest CP value at -0.5340864, and the 1 variable model had the lowest BIC value at -1210.518.

The backward selection method and stepwise selection method both resulted in a model with 4 predictors which were Density, Height, Hip and Thigh. Performing analysis on these methods we saw that the 4 variable model had the highest adjusted $R^2$ at 0.9979962, the 4 variable model had the lowest CP value at -2.37049448, and the 4 variable model had the lowest BIC value at -1213.929.

Since backward selection and stepwise selection methods coincided, the simpler 4 variable model was chosen. Therefore, the optimal multiple linear regression model came out to be: 
$y = 485.963-441.269(Density)-0.032(Height)+0.036(Hip)-0.041(Thigh)$ 
with an adjusted $R^2$ value of 0.998 and a test Mean Square Error ($MSE$) of 8.04. Interpreting our model we can say that a one unit increase in Density will reduce body fat percentage by 441.27 units, a one unit increase in Height will reduce body fat percentage by 0.03 units, a one unit increase in Hip will increase body fat percentage by 0.04 units, and a one unit increase in Thigh will reduce body fat percentage by 0.04 units. What is clear is that the more dense you are, the more likely you are to have a low body fat percentage. In other words, this means having more muscle than fat. One could hypothesize that being tall may help men have a lower body fat percentage because their basal metabolic rate is higher than the average male but they're likely consuming the same proportion sizes as the average male. This means the taller male is likely consuming the same amount of calories as the average male but is expending more which leads to less fat on the taller male.


The adjusted $R^2$ is a corrected goodness-of-fit (model accuracy) measure for linear models. A value of 1 would indicate a model that perfectly predicts body fat percentage. A value that is less than or equal to 0 indicates a model that has no predictive value. Therefore, this model is an almost perfect fit. A summary of the model is shown below in Figure (7). We see that all of our predictors are significant, which is indicated by levels of asterisks corresponding to how small its p-value is. In order of significance we see that Density is most significant to the model, followed by Hip, Thigh, and Height.


```
## 
## Call:
## lm(formula = Pct.BF ~ Density + Height + Hip + Thigh, data = trainingData)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -1.2150 -0.1308 -0.0248  0.0661  4.1718 
## 
## Coefficients:
##               Estimate Std. Error  t value Pr(>|t|)    
## (Intercept)  4.860e+02  2.306e+00  210.734  < 2e-16 ***
## Density     -4.413e+02  2.048e+00 -215.472  < 2e-16 ***
## Height      -3.198e-02  1.216e-02   -2.631 0.009201 ** 
## Hip          3.574e-02  9.626e-03    3.713 0.000267 ***
## Thigh       -4.050e-02  1.097e-02   -3.693 0.000288 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3607 on 194 degrees of freedom
## Multiple R-squared:  0.998,	Adjusted R-squared:  0.998 
## F-statistic: 2.466e+04 on 4 and 194 DF,  p-value: < 2.2e-16
```

Figure (7): Above we have the summary statistics for the 4 variable model $y = 485.963-441.269(Density)-0.032(Height)+0.036(Hip)-0.041(Thigh)$ along with the significance codes for the 4 variables. The smaller the p-value the more significant the predictor is to our model.



Running diagnostics on our model we see below in Figure (8) that the Residual vs Fitted plot we see that the residuals around the horizontal line are equally spread which is good but observations 74, 167, and 214 are notable. There really isn't a non-linear pattern in the plot either which is a good indication our model is performing well. On the Normal Q-Q plot the residuals follow the dashed line well although observations 74, 167, and 214 show up again. The Scale-Location plot doesn't seem to violate any assumptions as the line is fairly horizontal and the points are spread equally along the ranges of predictors. In the Residuals vs Leverage plot we see that observation 74 is almost within our Cook's distance lines. Observations 74, 167, and 214 may have been results of improper data entry or anomalies. Since we did not have enough information to remove them we didn't.

<img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-8-1.png" width="672" />

Figure (8): Displayed above is four plots showing the Residuals vs Fitted, Normal Q-Q, Scale-Location, and Residuals vs Leverage (Cook's Distance).


Now although the 4 variable model was the optimal multiple linear regression model, there is a simpler linear model that has almost identical accuracy. When performing LASSO regression we found that it zeroed out all predictors except for Density. The model is: $y = 488.19-444.40(Density)$ with an $MSE$ of 8.07 and an adjusted $R^2$ value of 0.9978. Interpreting this model, this means that a one unit increase in Density will reduce body fat percentage by 444.40 units. Since this model is much simpler than the 4 variable model and the $MSE$ was only greater by 0.03 (provides basically the same amount of accuracy), one could argue for the use of this model for its simplicity.


After performing KNN Regression on our data set with values of k = 3,5,7,9,11 we found that the best KNN model was k=7 and gave us a $MSE$ of 12.53, which is higher than our linear models. Performing Regression Trees led to a model with a $MSE$ of 7.02 and Bagging Random Forests produced a model with a $MSE$ of 5.63. Boosting Regression Trees with shrinkage led to a model with an $MSE$ of 4.70. This was the lowest $MSE$ out of all models in the study, which means it is the most accurate. The top five most important variables to the boosted regression tree model in order from most important were Density, Height, Chest, Hip, and Weight as seen below in Figure (9). Density, the most important feature accounts for 92.96% of the reduction to the loss function given this set of features.

<img src="Math_6333_Louis_Final_Project_RMarkdown_files/figure-html/unnamed-chunk-9-1.png" width="672" />

```
##             var     rel.inf
## Density Density 92.96473335
## Height   Height  1.86821015
## Chest     Chest  0.95571705
## Hip         Hip  0.90215926
## Weight   Weight  0.77970710
## Abdomen Abdomen  0.73686189
## Neck       Neck  0.54668395
## Knee       Knee  0.42963959
## Age         Age  0.16313791
## Bicep     Bicep  0.16016519
## Thigh     Thigh  0.13797496
## Wrist     Wrist  0.13131165
## Ankle     Ankle  0.12782360
## Forearm Forearm  0.09587435
## Waist     Waist  0.00000000
```

Figure (9): A table of our variables and there relative influence are displayed along with its bar chart. Density, the most important feature accounts for 92.96% of the reduction to the loss function given this set of features.


The downside of the Boosted Regression Trees is that they're black box models. Therefore, we can't interpret them like linear models and hence, they don't give you an explicit formula like linear regression does. Since this is the case, we recommend either the 4 variable or 1 variable linear models because of their ability to interpret the results, and their $MSE$ isn't much more than the boosted regression tree model. But, we recommend the boosted regression tree model if the main focus is accuracy since it does have the lowest $MSE$ out of all models.


## Discussion


In conclusion, we found the significant predictors to our linear regression models to be Density, Height, Hip, and Thigh, with Density being the most significant. For the most accurate model, the boosted regression tree, the most significant predictors in order were Density, Height, Chest, Hip, and Weight. In all three models we found Density to be the most important predictor with Height and Hip being common to the linear models and the boosted regression tree.


Knowing that these predictors contribute the most to body fat percentage prediction is important for prediction algorithms and for future studies. The results of this study may help reaffirm other studies that have found similar results. These results may also serve as a good indicator to future researchers for which measures should be included to predict body fat percentage. Unfortunately, density cannot be measured easily or quickly. It requires underwater submersion of the person to be calculated. Therefore it may be more helpful in future studies for the researchers to only include measurements which can be easily done by the participants.


Some limitations of the study were the number of observations, computing power, and time. A lot of the scatter plots generated didn't have any real obvious pattern to them. If we had more observations it would have been easier to identify trends and also would have made the algorithms much more precise. Computing power and time were limitations as there are a number of other regression algorithms and techniques we could have employed such as Neural Network Regression, Ridge Regression, XGBoost, and different combinations of parameters for regression trees.


## References

Harvard T.H. Chan School of Public Health. (2016, April 12). Measuring obesity. Obesity Prevention Source. Retrieved November 17, 2022, from <https://www.hsph.harvard.edu/obesity-prevention-source/obesity-definition/how-to-measure-body-fatness/>

Merrill Z, Chambers A, Cham R. Development and validation of body fat prediction models in American adults. Obes Sci Pract. 2020 Jan 15;6(2):189-195. doi: 10.1002/osp4.392. PMID: 32313677; PMCID: PMC7156815. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7156815/>

Fan Z, Chiong R, Hu Z, Keivanian F, Chiong F (2022) Body fat prediction through feature extraction based on anthropometric and laboratory measurements. PLOS ONE 17(2): e0263333. <https://doi.org/10.1371/journal.pone.0263333>

Stevens J, Ou FS, Cai J, Heymsfield SB, Truesdale KP. Prediction of percent body fat measurements in Americans 8 years and older. Int J Obes (Lond). 2016 Apr;40(4):587-94. doi: 10.1038/ijo.2015.231. Epub 2015 Nov 5. PMID: 26538187; PMCID: PMC5547817. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5547817/>

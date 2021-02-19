# SOEN6111-BIG-DATA project proposal
## ABSTRACT
   The model established in this project mainly discusses the housing price issue in India.
   
   The data for this project comes from kaggle, in which data from recent years is selected for this project. The data set provides 12 influencing factors. Therefore, when predicting the current Indian housing prices, this project should also consider the impact on Indian housing prices from the aspects of address, square, longitude, under construction, resale or not, etc. The main project process is data processing, visual analysis and modeling prediction.

## INTRODUCTION
### Context
  Buyers not only care about the size of the house (square feet), but various other factors play a key role in determining the price of the house/property. Finding the correct set of attributes to help understand buyer behavior can be very difficult. The data set has collected data from various real estate dealers in India. The data set provides 11 influencing factors and accurately predicts house prices through modeling.
  
### Objectives
  In real life, we often encounter the problem of classification and prediction. The target variable value is usually affected by multiple factors, and different factors have different effects on the target variable, so the weight will be different. Some factors are high, and some factors have a small weight. We usually use known factors to predict the value of the target variable.
  
  House prices are affected by many factors, such as address, area, and availability of immediate check-in. So we use big data analysis to get the trend of housing prices and the main factors affecting housing prices, and then we can be more familiar with the needs of buyers and the Indian housing market.

### The Problem to Solve
  1.What factors influence the price of houses?
  
  2.Deviding housing price data into three price levels(high-medium-low)，studing the impact of influencing factors on housing prices at different levels.
  
  3.The prediction of housing price in India in the future.
   
### Related Work
  The research on housing price forecasting in recent years found that dealing with housing price forecasting problems generally starts from the analysis of the influencing factors of housing price. Housing price analysis has a long history. Bensen used multiple regression method to predict housing price in 1997, and Malpezzi in 1999 Using time series cross-section regression to analyze housing prices, Song used GIS to extract influencing factors and then constructed a characteristic price model to analyze housing prices. The current trend is also the method of this project as follows:
  
  The general process is to first check whether the data is missing. If there is a missing value, the missing value must be processed first.Since there are many influencing factors, visual analysis will be carried out next. First, by calculating the correlation of the variables, properly draw the normal probability diagram, the distribution diagram of each quantitative variable, calculate the skewness of the quantitative data, and then perform quantitative feature analysis, specifically including analysis of variance or analysis of variance, and then perform related calculations. Since house prices are divided into three levels: high-medium-low, price breakdowns are needed.


## MATERIALS AND METHODS(400 words)
### The Dataset
The open dataset we utilised is from kaggle and the download page is linked below. https://www.kaggle.com/ishandutta/machine-hack-housing-price-prediction

This dataset has been collected across various property aggregators across India. The dataset is contained with 29451 rows and 12 columns, providing 12 influencing factors of the housing price, including 6 categorical features, 5 quantitative features.

 Variable Name             | Type   
 --------                  | :-----------:
 POST_BY                   | Categorical 
 UNDER_CONSTRUCTION        | Boolean 
 RERA                      | Boolean 
 BHK_NO.                   | Real number 
 BHK_OR_RK                 | Categorical
 SQUARE_FT                 | Real number
 READY_TO_MOVE             | Boolean
 RESALE                    | Boolean
 ADDRESS                   | Categorical
 LONGITUDE                 | Real number
 LATITUDE                  | Real number
 TARGET(PRICE_IN_LACS)     | Real number
 
 

## Technologies and Algorithms
Aiming to research the problems we proposed above, the design of our project includes four parts, which are data exploration, data cleaning, data analysis and modeling training and evaluation respectively.

### Data Explore
Data exploration is supposed to be completed at first by taking a quick inspection on the description of the dataset and visualizing all the 12 features to extract the features that needed to be cleaned in the following steps. 

The APIs used in this section are mainly Apache Spark library with Dataframe and Seaborn.

### Data Cleaning 
After completion of data exploration, we could have more information about the dataset. There is no missing value, and the address feature has 6899 distinct values which needed to be cleaned. To clean the address feature, we categorized the address based on different regions. In the following step, we need to transform the quantitative features making them follow the normal distribution, to implement a quick estimation of influence of categorical features on Price with ANOVA test, and encode the categorical features according to ordering based on the mean of Price. 

The APIs we mainly used in this section is Apache Spark Dataframe.

### Data Analysis
To solve the first problem, we need to figure out which features will have an important impact on the housing price by calculating the Spearman correlation between price and other features. The reason we choose the spearman correlation is because it picks up relationships between features even when they are non linear. The result of data analysis is illustrated by plotting heatmap and pairplot which is also useful to see how Price compares to each independent feature. 

To solve the second problem, all properties are divided in three price groups: low price, medium price and high price. Then means of quantitative features are compared to figure out the important influencing factors regarding different price groups. 

The APIs used in this section are mainly Spark Dataframe, SciPy and Seaborn. 

### Modeling Training and Evaluation
To solve the third problem, there is a need to build a machine learning model to do the prediction of housing price. After the data analysis, it is clear which features should be selected to train the model and which features should be dropped. The algorithm used in the training model is the decision tree which is able to capture non-linear relationships and is interpretable. In order to avoid the effect of data imbalance and take full advantage of our dataset, we will implement the KFold cross validation. The GridSearchCV will be implemented as well to search the best estimator for our problem. Finally, the R2 evaluation method would be used to evaluate the accuracy of the model. 

The APIs we mainly used in this section are Scikit-learn and Dataframe.

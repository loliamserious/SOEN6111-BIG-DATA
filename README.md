# SOEN6111-BIG-DATA Project GitHub repositoryURL
Project background: Understand the relationship between the production input parameters and results of a certain imported instrument in the factory, and conduct data prediction and sensitivity analysis.

Theoretical basis: According to the production process of a certain imported instrument in the factory, a one-year data set formed by capturing a set of data every 15 minutes for analysis. According to five data input sources and one output source, data modeling is performed, and then the latest The input factors in the actual production process of one month are tested and predicted, and the accuracy of the model establishment is proved by comparison with the true value curve.
# SOEN6111-BIG-DATA project proposal
ABSTRACT(100 words)
  The model established in this project mainly discusses the housing price issue in India.
  The data for this project comes from kaggle, in which data from recent years is selected for this project. The data set provides 12 influencing factors. Therefore, when predicting the current Indian housing prices, this project should also consider the impact on Indian housing prices from the aspects of address, square, longitude, under construction, resale or not, etc. The main project process is data processing, visual analysis and modeling prediction.

INTRODUCTION(300 words)
A.Context
   
  Buyers not only care about the size of the house (square feet), but various other factors play a key role in determining the price of the house/property. Finding the correct set of attributes to help understand buyer behavior can be very difficult. The data set has collected data from various real estate dealers in India. The data set provides 11 influencing factors and accurately predicts house prices through modeling.
B.Objectives
  In real life, we often encounter the problem of classification and prediction. The target variable value is usually affected by multiple factors, and different factors have different effects on the target variable, so the weight will be different. Some factors are high, and some factors have a small weight. We usually use known factors to predict the value of the target variable.

  House prices are affected by many factors, such as address, area, and availability of immediate check-in. So we use big data analysis to get the trend of housing prices and the main factors affecting housing prices, and then we can be more familiar with the needs of buyers and the Indian housing market


C.The Problem to Solve
                           1.What factors influence the price of houses?
                           2.The prediction of housing price in India in the future.
                           3.Deviding housing price data into three price levels(high-medium-low)，studing the impact of influencing factors on housing prices at different levels   
D.Related Work
  The research on housing price forecasting in recent years found that dealing with housing price forecasting problems generally starts from the analysis of the influencing factors of housing price. Housing price analysis has a long history. Bensen used multiple regression method to predict housing price in 1997, and Malpezzi in 1999 Using time series cross-section regression to analyze housing prices, Song used GIS to extract influencing factors and then constructed a characteristic price model to analyze housing prices. The current trend is also the method of this project as follows:
  The general process is to first check whether the data is missing. If there is a missing value, the missing value must be processed first.Since there are many influencing factors, visual analysis will be carried out next. First, by calculating the correlation of the variables, properly draw the normal probability diagram, the distribution diagram of each quantitative variable, calculate the skewness of the quantitative data, and then perform quantitative feature analysis, specifically including analysis of variance or analysis of variance, and then perform related calculations. Since house prices are divided into three levels: high-medium-low, price breakdowns are needed


MATERIALS AND METHODS(400 words)
The Dataset
The open dataset we utilised is from kaggle and the download page is linked below. https://www.kaggle.com/ishandutta/machine-hack-housing-price-prediction

This dataset has been collected across various property aggregators across India. The dataset is contained with 29451 rows and 12 columns, providing 12 influencing factors of the housing price, including 6 categorical features, 5 quantitative features and 1 string feature, which are category marking who has listed the property, under construction or not, rera approved or not, number of rooms, type of property, square feet, ready to move or not, category marking resale or not, address of the property, longitude of the property, latitude of the property and the price in LACs.
Technologies and Algorithms
Aiming to research the problems we proposed above, the design of our project includes four parts, which are data exploration, data cleaning, data analysis and modeling training and evaluation respectively.
Data Explore
Data exploration is supposed to be completed at first by taking a quick inspection on the description of the dataset using Apache Spark library with Dataframe API and visualizing all the 12 features by using Seaborn API to extract the features that needed to be cleaned in the following steps.

Data Cleaning 
After completion of data exploration, we could have more information about the dataset. There is no missing value, and the address feature has 6899 distinct values which needed to be cleaned. To clean the address feature, we categorized the address based on different regions using Apache Spark with Dataframe API. In the following step, we need to transform the quantitative features making them follow the normal distribution, to implement a quick estimation of influence of categorical features on Price with ANOVA test, and encode the categorical features according to ordering based on the mean of Price.

Data Analysis
To solve the first problem, we need to figure out which features will have an important impact on the housing price by calculating the Spearman correlation between price and other features. The reason we choose the spearman correlation is because it picks up relatiusing Dataframe and SciPy API. The result of data analysis is illustrated by using Seaborn API. 

    
Modeling Training and Evaluation
To solve the second problem, we decided to apply the decision tree algorithm which is able to capture non-linear relationships and interpretable to build the model and to predict the price of property. In order to avoid the effect of data imbalance and take full advantage of our dataset, we will implement the KFold cross validation. The GridSearchCV will be implemented as well to search the best estimator for our problem. Finally, the R2 evaluation method would be used to evaluate the accuracy of the model. The APIs we mainly used in this section are Scikit-learn and Dataframe.

from DataExploration import *
from DataPreprocessing import *
from DataAnalysis import *
from RegressionModels import *





spark = init_spark()

# Codes for data exploration, pre-processing and analysis

df_dataset = spark.read.csv("./data/Train.csv", header="true", inferSchema="true")
df_dataset.repartition(1000)
# Handle missing values
check_missing_value(df_dataset)
df_dataset = drop_exception_data(df_dataset)

# Data Visualization
visualize_post_by(df_dataset)
visualize_under_construction(df_dataset)
visualize_rera(df_dataset)
visualize_bhk_no(df_dataset)
visualize_bhk_or_rk(df_dataset)
visualize_square_ft(df_dataset)
visualize_ready_to_move(df_dataset)
visualize_resale(df_dataset)

# Data pre-processing and encoding
df_dataset = address_to_state(spark, df_dataset)
visualize_state(df_dataset)
# Perform log transformation on quantitative features of data before feeding the training model
df_dataset = log_transform(df_dataset, ['BHK_NO','SQUARE_FT','TARGET(PRICE_IN_LACS)'])
df_dataset = one_hot_encoding(df_dataset, 'POSTED_BY')
df_dataset = one_hot_encoding(df_dataset, 'BHK_OR_RK')
df_train = one_hot_encoding(df_dataset, 'state')
df_plot = target_mean_encode(df_dataset,'state','TARGET(PRICE_IN_LACS)')

# Data analysis
spearman_corr(df_plot)
pearson_corr_heatmap(spark, df_plot)
price_segments(df_plot)
# Drop the features with high similarity with other features
df_dataset = df_train.drop('READY_TO_MOVE','RK','Owner')



# Codes for training models and evaluating models

# Split the data into training and test sets (20% held out for testing)
(X_train, X_test) = df_dataset.randomSplit([0.8, 0.2])

# Train random forest model and evaluate the model
rf_model = train_rf_model(X_train)
evaluate_model(X_test, rf_model, 'Random Forest Regression')

# Train gbt model and evaluate the model
gbt_model = train_gbt_model(X_train)
evaluate_model(X_test, gbt_model, 'GBT Regression')

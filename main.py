from DataExploration import *
from DataCleaning import *
from DataAnalysis import *
from RandomForestRegression import *
from pyspark.sql.functions import col





spark = init_spark()

#code for data analysis and data cleaning

df_dataset = spark.read.csv("./data/Train.csv", header="true", inferSchema="true")
# check_missing_value(df_dataset)
df_dataset = drop_exception_data(df_dataset)

#visualize_post_by(df_dataset)
#visualize_under_construction(df_dataset)
#visualize_rera(df_dataset)
#visualize_bhk_no(df_dataset)
#visualize_bhk_or_rk(df_dataset)
#visualize_square_ft(df_dataset)
#visualize_ready_to_move(df_dataset)
#visualize_resale(df_dataset)

df_dataset = address_to_state(spark, df_dataset)
# visualize_state(df_dataset)
df_dataset = target_mean_encode(df_dataset, ['POSTED_BY', 'state'], 'TARGET(PRICE_IN_LACS)')
# spearman_corr(df_dataset)
# df_corr_matrix = pearson_corr_heatmap(spark, df_dataset)
# price_segments(df_dataset) #change price range
# drop_similar_feature(df_corr_matrix)
df_dataset = df_dataset.drop('UNDER_CONSTRUCTION', 'READY_TO_MOVE')
df_dataset.coalesce(1).write.option('header', 'true').csv("./data/train_cleaned")


# code for training model and evaluating model
#train_model(spark)
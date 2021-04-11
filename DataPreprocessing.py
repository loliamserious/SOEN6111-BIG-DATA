from DataExploration import *
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum,split
from sklearn.preprocessing import MinMaxScaler



def drop_exception_data(dataset):
    """
    This function is used to drop the exception of dataset.
    For example, some houses with extreme large square ft, like 50000 sqt, need to be dropped as exception.
    """
    df_expection = dataset.filter((F.col('TARGET(PRICE_IN_LACS)') > 2000.0) | (F.col('SQUARE_FT') > 30000.0))
    df_dataset = dataset.exceptAll(df_expection)
    return df_dataset


def address_to_state(spark, dataset):
    """
    This function is used to categorize variable 'ADDRESS' into different state based on its city.
    The format of address is 'street, city', and each city belongs to a state of India.
    There are 33 states in India, so we need to categorize all address into 33 categories and form a new variable 'state'.
    """
    df_indian_city1 = spark.read.csv("./data/india_cities_states_feb_2015.csv", header="true", inferSchema="true")
    df_indian_city2 = spark.read.csv("./data/Indian Cities Database.csv", header="true", inferSchema="true")
    df_indian_city3 = spark.read.csv("./data/in.csv", header="true", inferSchema="true")
    df_indian_city = df_indian_city1.union(
        df_indian_city2.select('City', 'State').withColumnRenamed('State', 'state').withColumnRenamed('City', 'city')) \
        .union(df_indian_city3.select('city', 'admin_name').withColumnRenamed('admin_name', 'state')).distinct()
    dataset = dataset.withColumn('split_address', split(col('ADDRESS'), ',')).withColumn('ADDRESS', col(
        'split_address').getItem(1)) \
        .drop('split_address', 'LONGITUDE', 'LATITUDE')
    df_dataset = dataset.join(df_indian_city, dataset['ADDRESS'] == df_indian_city['city'], 'left_outer')
    df_dataset = df_dataset.withColumn('state',
                                       F.when(col('state').isNull(), 'Other').otherwise(col('state'))).drop('city',
                                                                                                            'ADDRESS')
    return df_dataset


def one_hot_encoding(dataset, feature):
    """
    This function is used to encode the categorical variable with category less than 3
    Use the one hot encoding method.
    """
    feature_column = dataset.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
    ohe_feature = [F.when(F.col(feature) == cat, 1).otherwise(0).alias(str(cat)) for cat in feature_column]
    df_dataset = dataset.select(dataset.columns+ohe_feature).drop(feature)
    return df_dataset


def target_mean_encode(dataset, feature, target):
    """
    This function is used to encode the categorical variable with category more than 5
    Use the target mean encoding method
    """
    target_encoded_columns_list = []
    means = dataset.groupby(F.col(feature)).agg(F.mean(target).alias(f"{feature}_mean_encoding"))
    dict_ = means.toPandas().to_dict()
    target_encoded_columns = [F.when(F.col(feature) == v, encoder)
                              for v, encoder in
                              zip(dict_[feature].values(), dict_[f"{feature}_mean_encoding"].values())]
    target_encoded_columns_list.append(F.coalesce(*target_encoded_columns).alias(f"{feature}_mean_encoding"))
    dataset = dataset.withColumn(f"{feature}_mean_encoding", *target_encoded_columns_list).drop(feature)
    column_list = dataset.drop('TARGET(PRICE_IN_LACS)','RK').columns + ['TARGET(PRICE_IN_LACS)']
    dataset = dataset.select(column_list)
    for c in dataset.columns:
        dataset = dataset.withColumn(c, col(c).cast('float'))
    return dataset


def log_transform(dataset, features):
    """
    This function is used to do log transformation on quantitative features to make them follow normal distribution
    """
    for feature in features:
        dataset = dataset.withColumn(feature, F.log(feature))
    return dataset


def scaling(spark, dataset, features):
    """
    This function is used to scale each feature to a specific range [0, 1]
    :param spark: spark
    :param dataset: spark dataframe
    :param features: quantitative features need to be scaled
    :return: spark dataframe
    """
    spark = init_spark()
    scaler = MinMaxScaler()
    pdf_dataset = dataset.toPandas()
    pdf_dataset[features] = scaler.fit_transform(pdf_dataset[features])
    df_dataset = spark.createDataFrame(pdf_dataset)
    return df_dataset

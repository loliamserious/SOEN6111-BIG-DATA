import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum,split,udf





def drop_exception_data(dataset):
    df_expection = dataset.filter((F.col('TARGET(PRICE_IN_LACS)') > 2000.0) | (F.col('SQUARE_FT') > 30000.0))
    df_dataset = dataset.exceptAll(df_expection)
    return df_dataset


def address_to_state(spark, dataset):
    '''
    rdd_indian_state = indian_city.rdd
    rdd_dataset = dataset.select('LONGITUDE', 'LATITUDE').rdd
    rdd_join = rdd_dataset.cartesian(rdd_indian_state).map(
        lambda x: (x[0][0], x[0][1], x[1][1], x[0][0] - x[1][2], x[0][1] - x[1][3])) \
        .filter(lambda x: (0 < x[3] < 3 or -3 < x[3] < 0) and (0 < x[4] < 3 or -3 < x[4] < 0)).map(
        lambda x: Row((x[0], x[1]), x)).reduceByKey(lambda x, y: min_longitude_row(x, y)) \
        .map(lambda x: (x[1][0], x[1][1], x[1][2]))
    df_join = spark.createDataFrame(rdd_join)
    df_dataset = dataset.drop('BHK_OR_RK', 'ADDRESS').join(df_join, (dataset.LONGITUDE == df_join._1) & (
                dataset.LATITUDE == df_join._2)).drop('_1', '_2', 'LONGITUDE', 'LATITUDE') \
        .withColumnRenamed('_3', 'STATE')
    '''
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
                                                                                                            'ADDRESS',
                                                                                                            'BHK_OR_RK')
    return df_dataset


def target_mean_encode(dataset, features, target):
    '''
    feature_column = dataset.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
    ohe_feature = [F.when(F.col(feature) == cat, 1).otherwise(0).alias(str(cat)) for cat in feature_column]
    df_dataset = dataset.select(dataset.columns+ohe_feature).drop(feature)
    '''
    for f in features:
        target_encoded_columns_list = []
        means = dataset.groupby(F.col(f)).agg(F.mean(target).alias(f"{f}_mean_encoding"))
        dict_ = means.toPandas().to_dict()
        target_encoded_columns = [F.when(F.col(f) == v, encoder)
                                  for v, encoder in zip(dict_[f].values(), dict_[f"{f}_mean_encoding"].values())]
        target_encoded_columns_list.append(F.coalesce(*target_encoded_columns).alias(f"{f}_mean_encoding"))
        dataset = dataset.withColumn(f"{f}_mean_encoding", *target_encoded_columns_list).drop(f)
    column_list = dataset.drop('TARGET(PRICE_IN_LACS)').columns + ['TARGET(PRICE_IN_LACS)']
    dataset = dataset.select(column_list)
    for c in dataset.columns:
        dataset = dataset.withColumn(c, col(c).cast('float'))
    return dataset
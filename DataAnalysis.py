from pyspark.sql import SparkSession,Row
from pyspark.sql.functions import col, sum,split,udf
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def spearman_corr(dataset):
    """
    This function is used to analyze the spearman correlation between each variable and target variable.
    """
    df_corr = pd.DataFrame()
    df_corr['features'] = dataset.columns
    list_spearman_corr = []
    for c in dataset.columns:
        list_spearman_corr.append(dataset.stat.corr(c, 'TARGET(PRICE_IN_LACS)'))
    df_corr['spearman'] = list_spearman_corr
    df_corr = df_corr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25 * len(list_spearman_corr)))
    sns.barplot(data=df_corr, y='features', x='spearman', orient='h')
    plt.savefig('spearman_corr_barplot.png', bbox_inches='tight')
    plt.show()


def pearson_corr_heatmap(spark,dataset):
    """
    This function is used to analyze the pearson correlation between each variable including the target variable.
    Generally to reduce confunding only variables uncorrelated with each other should be added to regression models.
    """
    assembler = VectorAssembler(inputCols=dataset.columns, outputCol='features', handleInvalid='keep')
    assembled = assembler.transform(dataset)
    pearson_corr = Correlation.corr(assembled, 'features')
    corr_list = pearson_corr.head()[0].toArray().tolist()
    pearson_corr_df = spark.createDataFrame(corr_list)
    mapping = dict(zip(pearson_corr_df.columns, dataset.columns))
    pearson_corr_df = pearson_corr_df.select([col(c).alias(mapping.get(c, c)) for c in pearson_corr_df.columns])
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr_df.toPandas(), annot=True, yticklabels=pearson_corr_df.columns)
    plt.savefig('pearson_corr_heatmap.png', bbox_inches='tight')
    plt.show()


def price_segments(dataset):
    """
    This function is used to analyze if the correlations shift with change of house price.
    First, dividing house price into 3 segments: low price, standard price, high price.
    Then means of quantitative variables are compared among three segments.
    Finally, plotting 3 figures to illustrate results.
    """
    df_low_price = dataset.filter(dataset['TARGET(PRICE_IN_LACS)'] <= 1.5)
    df_standard_price = dataset.filter(
        (1.5 < dataset['TARGET(PRICE_IN_LACS)']) & (dataset['TARGET(PRICE_IN_LACS)'] <= 5.5))
    df_high_price = dataset.filter(dataset['TARGET(PRICE_IN_LACS)'] > 5.5)

    list_diff_low_stand = []
    list_diff_low_high = []
    list_diff_stand_high = []
    for c in dataset.drop('TARGET(PRICE_IN_LACS)').columns:
        rdd_low = df_low_price.select(col(c)).rdd
        avg_low = rdd_low.map(lambda x: x[0]).reduce(lambda x, y: x + y) / rdd_low.count()
        rdd_stand = df_standard_price.select(col(c)).rdd
        avg_stand = rdd_stand.map(lambda x: x[0]).reduce(lambda x, y: x + y) / rdd_stand.count()
        rdd_high = df_high_price.select(col(c)).rdd
        avg_high = rdd_high.map(lambda x: x[0]).reduce(lambda x, y: x + y) / rdd_high.count()
        list_diff_low_stand.append((avg_low - avg_stand) / avg_stand)
        list_diff_low_high.append((avg_low - avg_high) / avg_high)
        list_diff_stand_high.append((avg_stand - avg_high) / avg_high)

    diff_low_stand = pd.DataFrame()
    diff_low_stand['feature'] = dataset.drop('TARGET(PRICE_IN_LACS)').columns
    diff_low_stand['difference'] = list_diff_low_stand
    diff_low_high = pd.DataFrame()
    diff_low_high['feature'] = dataset.drop('TARGET(PRICE_IN_LACS)').columns
    diff_low_high['difference'] = list_diff_low_high
    diff_stand_high = pd.DataFrame()
    diff_stand_high['feature'] = dataset.drop('TARGET(PRICE_IN_LACS)').columns
    diff_stand_high['difference'] = list_diff_stand_high
    plt.figure()
    sns.barplot(data=diff_low_stand, x='feature', y='difference')
    plt.title('Low Price Vs Standard Price')
    plt.xticks(rotation=90)
    plt.savefig('low_stand_price.png', bbox_inches='tight')
    plt.figure()
    sns.barplot(data=diff_low_high, x='feature', y='difference')
    plt.title('Low Price Vs High Price')
    plt.xticks(rotation=90)
    plt.savefig('low_high_price.png', bbox_inches='tight')
    plt.figure()
    sns.barplot(data=diff_stand_high, x='feature', y='difference')
    plt.title('High Price Vs Standard Price')
    plt.xticks(rotation=90)
    plt.savefig('stand_high_price.png', bbox_inches='tight')
    plt.show()

from pyspark.sql import SparkSession,Row
from pyspark.sql.functions import col, sum,split,udf
import pyspark.sql.functions as F
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd





def init_spark():
    spark = (
        SparkSession.builder.appName("Housing Price Anslysis")
            .config("spark.default.parallelism", 300)
            .config("spark.network.timeout", 10000000)
            .config("spark.executor.heartbeatInterval", 1000000)
            .getOrCreate()
    )
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    return spark


def check_missing_value(dataset):
    """
    This function check if there are missing values in the dataset.
    """
    dataset.printSchema()
    dataset.describe().show()
    dataset.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in dataset.columns)).show()


def visualize_post_by(dataset):
    """
    This function is used to visualize the distribution of the variable 'POSTED_BY'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_posted_by = dataset.groupBy('POSTED_BY').count().toPandas()
    df_posted_by_price = dataset.select('POSTED_BY', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 10))
    sns.barplot(x='POSTED_BY', y='count', data=df_posted_by, ax=ax1)
    ax1.set_title('POSTED_BY Distribution')
    ax1.set_xlabel('Variable=POSTED_BY')
    ax1.set_ylabel('Count')
    sns.boxplot(x='POSTED_BY', y='TARGET(PRICE_IN_LACS)', data=df_posted_by_price, ax=ax2)
    ax2.set_title('POSTED_BY Vs Housing Price')
    ax2.set_xlabel('Variable=POSTED_BY')
    ax2.set_ylabel('Housing Price')
    plt.savefig('posted_by_bar_boxplot.png', bbox_inches='tight')
    plt.show()


def visualize_under_construction(dataset):
    """
    This function is used to visualize the distribution of the variable 'UNDER_CONSTRUCTION'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_under_construction = dataset.groupBy('UNDER_CONSTRUCTION').count().toPandas()
    df_under_construction_price = dataset.select('UNDER_CONSTRUCTION', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 10))
    sns.barplot(x='UNDER_CONSTRUCTION', y='count', data=df_under_construction, ax=ax1)
    ax1.set_title('UNDER_CONSTRUCTION Distribution')
    ax1.set_xlabel('Variable=UNDER_CONSTRUCTION')
    ax1.set_ylabel('Count')
    sns.boxplot(x='UNDER_CONSTRUCTION', y='TARGET(PRICE_IN_LACS)', data=df_under_construction_price, ax=ax2)
    ax2.set_title('UNDER_CONSTRUCTION Vs Housing Price')
    ax2.set_xlabel('Variable=UNDER_CONSTRUCTION')
    ax2.set_ylabel('Housing Price')
    plt.savefig('under_const_bar_boxplot.png', bbox_inches='tight')
    plt.show()


def visualize_rera(dataset):
    """
    This function is used to visualize the distribution of the variable 'RERA'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_rera = dataset.groupBy('RERA').count().toPandas()
    df_rera_price = dataset.select('RERA', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 10))
    sns.barplot(x='RERA', y='count', data=df_rera, ax=ax1)
    ax1.set_title('RERA Distribution')
    ax1.set_xlabel('Variable=RERA')
    ax1.set_ylabel('Count')
    sns.boxplot(x='RERA', y='TARGET(PRICE_IN_LACS)', data=df_rera_price, ax=ax2)
    ax2.set_title('RERA Vs Housing Price')
    ax2.set_xlabel('Variable=RERA')
    ax2.set_ylabel('Housing Price')
    plt.savefig('rera_bar_boxplot.png', bbox_inches='tight')
    plt.show()


def visualize_bhk_no(dataset):
    """
    This function is used to visualize the distribution of the variable 'BHK_NO'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_bhk_no_price = dataset.select('BHK_NO', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 16))
    sns.countplot(y='BHK_NO', data=df_bhk_no_price, ax=ax1)
    ax1.set_title('BHK_NO Distribution')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Variable=BHK_NO')
    sns.boxplot(x='BHK_NO', y='TARGET(PRICE_IN_LACS)', data=df_bhk_no_price, ax=ax2)
    ax2.set_title('BHK_NO Vs Housing Price')
    ax2.set_xlabel('Variable=BHK_NO')
    ax2.set_ylabel('Housing Price')
    plt.savefig('bhk_no_count_boxplot.png', bbox_inches='tight')
    plt.show()


def visualize_bhk_or_rk(dataset):
    """
    This function is used to visualize the distribution of the variable 'BHK_OR_RK'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_bhk_rk = dataset.groupBy('BHK_OR_RK').count().toPandas()
    plt.figure(figsize=(5, 5))
    sns.barplot(x='BHK_OR_RK', y='count', data=df_bhk_rk)
    plt.title('BHK_OR_RK Distribution')
    plt.xlabel('BHK_OR_RK')
    plt.ylabel('Count')
    plt.savefig('bhk_rk_barplot.png', bbox_inches='tight')
    plt.show()


def visualize_square_ft(dataset):
    """
    This function is used to visualize the distribution of the variable 'SQUARE_FT'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_square_ft = dataset.select('SQUARE_FT', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12))
    sns.distplot(df_square_ft['SQUARE_FT'], kde=True, ax=ax1)
    ax1.set_title('SQUARE_FT Distribution')
    ax1.set_xlabel('Variable=SQUARE_FT')
    sns.regplot(x='SQUARE_FT', y='TARGET(PRICE_IN_LACS)', data=df_square_ft, ax=ax2)
    ax2.set_title('SQUARE_FT Vs Housing Price')
    ax2.set_xlabel('Variable=SQUARE_FT')
    ax2.set_ylabel('Housing Price')
    plt.savefig('square_ft_hist_regplot.png', bbox_inches='tight')
    plt.show()


def visualize_ready_to_move(dataset):
    """
    This function is used to visualize the distribution of the variable 'READY_TO_MOVE'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_ready_move = dataset.groupBy('READY_TO_MOVE').count().toPandas()
    df_ready_move_price = dataset.select('READY_TO_MOVE', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 10))
    sns.barplot(x='READY_TO_MOVE', y='count', data=df_ready_move, ax=ax1)
    ax1.set_title('READY_TO_MOVE Distribution')
    ax1.set_xlabel('Variable=READY_TO_MOVE')
    ax1.set_ylabel('Count')
    sns.boxplot(x='READY_TO_MOVE', y='TARGET(PRICE_IN_LACS)', data=df_ready_move_price, ax=ax2)
    ax2.set_title('READY_TO_MOVE Vs Housing Price')
    ax2.set_xlabel('Variable=READY_TO_MOVE')
    ax2.set_ylabel('Housing Price')
    plt.savefig('ready_to_move_bar_boxplot.png', bbox_inches='tight')
    plt.show()


def visualize_resale(dataset):
    """
    This function is used to visualize the distribution of the variable 'RESALE'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_resale = dataset.groupBy('RESALE').count().toPandas()
    df_resale_price = dataset.select('RESALE', 'TARGET(PRICE_IN_LACS)').toPandas()
    f, [ax1, ax2] = plt.subplots(2, 1, figsize=(5, 10))
    sns.barplot(x='RESALE', y='count', data=df_resale, ax=ax1)
    ax1.set_title('RESALE Distribution')
    ax1.set_xlabel('Variable=RESALE')
    ax1.set_ylabel('Count')
    sns.boxplot(x='RESALE', y='TARGET(PRICE_IN_LACS)', data=df_resale_price, ax=ax2)
    ax2.set_title('RESALE Vs Housing Price')
    ax2.set_xlabel('Variable=RESALE')
    ax2.set_ylabel('Housing Price')
    plt.savefig('resale_barplot_boxplot.png', bbox_inches='tight')
    plt.show()


def visualize_state(dataset):
    """
    This function is used to visualize the distribution of the variable 'state'
    and the statistical metrics on variable 'TARGET(PRICE_IN_LACS)'
    """
    df_state_price = dataset.select('state', 'TARGET(PRICE_IN_LACS)').toPandas()
    plt.figure(figsize=(32, 10))
    sns.boxplot(x='state', y='TARGET(PRICE_IN_LACS)', data=df_state_price)
    plt.title('State Distribution')
    plt.xlabel('State')
    plt.ylabel('Housing Price')
    plt.tight_layout()
    plt.savefig('state_boxplot.png')
    plt.show()












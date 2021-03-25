from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col





def train_model(spark):
    df_data = spark.read.csv("./data/train_clean.csv", header="true", inferSchema="true").distinct()
    X_coloums = df_data.columns[0:-1]

    # Assemble each row of features into a Vector
    assembler = VectorAssembler(inputCols=X_coloums, outputCol='vectoredFeatures')

    # Split the data into training and test sets (20% held out for testing)
    (X_train, X_test) = df_data.randomSplit([0.8, 0.2])

    # Train a RandomForest Regression model.
    rf = RandomForestRegressor(featuresCol="vectoredFeatures",labelCol='TARGET(PRICE_IN_LACS)',numTrees=50,maxDepth=10)

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # Train model.
    model = pipeline.fit(X_train)

    rfModel = model.stages[1]
    print(rfModel)  # summary only

    # Evaluate a model and plot the evaluation result
    evaluate_model(X_test, model)

    # K-fold and ParamGridMap API:Cross-Validation
    # Gradient-Boosted algorithm comparision


def evaluate_model(X_test, model):
    # Make predictions.
    predictions = model.transform(X_test)

    # Select example rows to display.
    predictions.select("vectoredFeatures", "TARGET(PRICE_IN_LACS)", "prediction").show()

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="TARGET(PRICE_IN_LACS)", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    # accuracy
    # F1 score
    # precision and recall
    # plot
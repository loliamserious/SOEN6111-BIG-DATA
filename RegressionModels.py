import math
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import numpy as np



def train_rf_model(X_train):

    # Select input coloums for feature vectors
    X_coloums = X_train.columns[0:-1]

    # Assemble each row of features into a Vector
    assembler = VectorAssembler(inputCols=X_coloums, outputCol='vectoredFeatures')

    # Train a RandomForest Regression model.
    rf = RandomForestRegressor(featuresCol="vectoredFeatures",labelCol='TARGET(PRICE_IN_LACS)', numTrees=100, seed=42)

    # Chain indexer and forest in a Pipeline
    pipeline_rf = Pipeline(stages=[assembler, rf])

    # 5-fold and ParamGridMap API:Cross-Validation
    paramGrid_rf = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [8,10,12,15]) \
        .build()

    crossval_rf = CrossValidator(estimator=pipeline_rf,
                              estimatorParamMaps=paramGrid_rf,
                              evaluator=RegressionEvaluator(labelCol="TARGET(PRICE_IN_LACS)",
                                                            predictionCol="prediction", metricName="r2"),
                              numFolds=5)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel_rf = crossval_rf.fit(X_train)

    # Print the average metrics of models on different value of maxDepth
    print('Average R2 on train data = %g' % cvModel_rf.avgMetrics[0])

    # Print the best parameter found by grid search
    best_model_rf = cvModel_rf.bestModel
    best_model_rf_param = {param[0].name: param[1] for param in best_model_rf.stages[1].extractParamMap().items()}
    print('The best parameter maxDepth = %s' % str(best_model_rf_param.get('maxDepth')))

    return best_model_rf



def train_gbt_model(X_train):
    # Select input coloums for feature vectors
    X_coloums = X_train.columns[0:-1]

    # Assemble each row of features into a Vector
    assembler = VectorAssembler(inputCols=X_coloums, outputCol='vectoredFeatures')

    # Gradient-Boosted algorithm comparision
    # Train a GradientBoostedTrees model.
    gbt = GBTRegressor(featuresCol="vectoredFeatures", labelCol='TARGET(PRICE_IN_LACS)', maxIter=100, seed=42)

    # Chain indexer and gbt boosted tree in a Pipeline
    pipeline_gbt = Pipeline(stages=[assembler, gbt])

    # 5-fold and ParamGridMap API:Cross-Validation
    paramGrid_gbt = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [8,10,12,15]) \
        .build()

    crossval_gbt = CrossValidator(estimator=pipeline_gbt,
                                 estimatorParamMaps=paramGrid_gbt,
                                 evaluator=RegressionEvaluator(labelCol="TARGET(PRICE_IN_LACS)",
                                                               predictionCol="prediction", metricName="r2"),
                                 numFolds=5)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel_gbt = crossval_gbt.fit(X_train)

    # Print the average metrics of models on different value of maxDepth
    print('Average R2 on train data = %g' % cvModel_gbt.avgMetrics[0])

    # Print the best parameter found by grid search
    best_model_gbt = cvModel_gbt.bestModel
    best_model_gbt_param = {param[0].name: param[1] for param in best_model_gbt.stages[1].extractParamMap().items()}
    print('The best parameter maxDepth = %s' % str(best_model_gbt_param.get('maxDepth')))

    return best_model_gbt



def evaluate_model(X_test, model, picname):

    # Make predictions.
    predictions = model.transform(X_test)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="TARGET(PRICE_IN_LACS)", predictionCol="prediction", metricName="rmse")

    evaluator1 = RegressionEvaluator(
        labelCol="TARGET(PRICE_IN_LACS)", predictionCol="prediction", metricName="mse")

    evaluator2 = RegressionEvaluator(
        labelCol="TARGET(PRICE_IN_LACS)", predictionCol="prediction", metricName="r2")

    rmse = evaluator.evaluate(predictions)
    mse = evaluator1.evaluate(predictions)
    r2=evaluator2.evaluate(predictions)
    
    print("Root Mean Squared Error (RMSE) on test data best result = %g" % rmse)
    print("Mean Squared Error (MSE) on test data best result = %g" % mse)
    print("R2 on test data best result = %g" % r2)
    
    # plot
    plt.figure()
    x=np.array(list(range(1,predictions.select("TARGET(PRICE_IN_LACS)").count()+1)))
    plt.scatter(x, predictions.select("TARGET(PRICE_IN_LACS)").toPandas(), s=20, edgecolor="black",
                c="darkorange", label="original")
    plt.plot(x, predictions.select("prediction").toPandas(), color="cornflowerblue",
             label="prediction", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title(picname)
    plt.legend()
    plt.savefig(picname+'.png', bbox_inches='tight')
    plt.show()
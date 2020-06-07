"""
    Code taken from: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
"""
from pyspark.sql.session import SparkSession


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os


spark = SparkSession.builder.getOrCreate()
SPARK_EXTRACTED_LOCATION = "/home/anurag/Spark/spark-2.4.0-bin-hadoop2.7"

# Load and parse the data file, converting it to a DataFrame.
sample_data_location = os.path.join(SPARK_EXTRACTED_LOCATION, "data/mllib/sample_libsvm_data.txt")

# libsvm Data format: http://apache-spark-user-list.1001560.n3.nabble.com/Description-of-data-file-sample-libsvm-data-txt-td25832.html
# Sparse matrix format: <label> <index1>:<value1> <index2>:<value2> ...
# If the index is missing - means the value is 0
data = spark.read.format("libsvm").load(sample_data_location)

# 
# data.show(n=10, truncate=False)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5, truncate=False)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

print("Saving the model")
model.save("output")

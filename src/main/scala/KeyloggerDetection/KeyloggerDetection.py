from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

INFECTED = "infected"
PROTOCOL = "protocol"
PROTOCOL_VEC = "protocol_vec"
pipelineSelectedCols = ["protocol", "features_raw"]  # Update with your actual columns

def load_df(spark):
    df = spark.read.parquet("./data/processed/processed.parquet")
    return df

def get_model(model_name):
    if model_name == "LogisticRegression":
        model = LogisticRegression(labelCol=INFECTED, maxIter=10, regParam=0.3, elasticNetParam=0.8)
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(labelCol=INFECTED)
    else:
        raise ValueError("Invalid model name")
    return model

def create_pipeline(model_name):
    protocol_encoder = OneHotEncoder(inputCols=[PROTOCOL], outputCols=[PROTOCOL_VEC])
    assembler = VectorAssembler(inputCols=pipelineSelectedCols, outputCol="features")
    model = get_model(model_name)
    pipeline = Pipeline(stages=[protocol_encoder, assembler, model])
    return pipeline

def print_metrics(df):
    TP = df.filter((col(INFECTED) == col("prediction")) & (col(INFECTED) == 1)).count()
    TN = df.filter((col(INFECTED) == col("prediction")) & (col(INFECTED) == 0)).count()
    FP = df.filter((col(INFECTED) != col("prediction")) & (col(INFECTED) == 1)).count()
    FN = df.filter((col(INFECTED) != col("prediction")) & (col(INFECTED) == 0)).count()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + TN)
    F1 = (precision * recall) / (precision + recall)
    print(f"True Positives: {TP}\nTrue Negatives: {TN}\nFalse Positives: {FP}\nFalse Negatives: {FN}\n")
    print(f"=== Confusion Matrix ===\n|    {TP}     |     {FP}    |\n|    {FN}     |     {TN}    |")
    print(f"\nAccuracy = {accuracy}")
    print(f"\nPrecision = {precision}")
    print(f"\nRecall = {recall}")
    print(f"\nF1 = {F1}")

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("KeyloggerDetection") \
        .master("local[*]") \
        .getOrCreate()

    processed_df = load_df(spark)

    print("============ Training Model =============")
    # Update model name as needed
    pipeline = create_pipeline("LogisticRegression")
    fitted_model = pipeline.fit(processed_df)

    # Uncomment below to perform predictions and print metrics
    # predictions = fitted_model.transform(processed_df)
    # print_metrics(predictions)

    spark.stop()

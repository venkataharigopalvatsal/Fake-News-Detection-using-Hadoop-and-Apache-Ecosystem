import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object FakeNewsClassifierApp {
  def main(args: Array[String]): Unit = {

    // Initialize SparkSession
    val spark = SparkSession.builder
      .appName("FakeNewsClassifier")
      .master("local[*]") // use local[*] or remove for cluster mode
      .getOrCreate()

    import spark.implicits._

    // Read CSV from HDFS
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("hdfs://localhost:9000/charan/augmented_news_1000MB.csv")  // HDFS path

    // Show sample
    df.show(5)

    // Drop rows where text is null
    val dfClean = df.na.drop(Seq("text"))

    // If you have labels (e.g., real/fake) as a column "label", otherwise create dummy
    // For demo, let's assume a column "label" exists: 0=real, 1=fake
    // val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex").fit(dfClean)
    
    // Tokenize text
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    
    // Remove stopwords
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")
    
    // TF
    val hashingTF = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(10000)
    
    // IDF
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    
    // Logistic Regression classifier
    val lr = new LogisticRegression()
      .setLabelCol("label")  // replace with your actual label column
      .setFeaturesCol("features")
      .setMaxIter(20)
    
    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, lr))
    
    // Train-test split
    val Array(train, test) = dfClean.randomSplit(Array(0.8, 0.2), seed=42)
    
    // Train model
    val model = pipeline.fit(train)
    
    // Make predictions
    val predictions = model.transform(test)
    
    // Evaluate
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Accuracy = $accuracy")

    spark.stop()
  }
}

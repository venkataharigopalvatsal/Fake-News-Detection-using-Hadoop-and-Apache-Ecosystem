import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, HashingTF, IDF}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

object FakeNewsClassifierLocal {
  def main(args: Array[String]): Unit = {

    // ---------- Spark Session ----------
    val spark = SparkSession.builder()
      .appName("FakeNewsClassifierLocal")
      .master("local[*]") // Use all local cores
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    import spark.implicits._

    // ---------- Load Data ----------
    val fakeDF = spark.read.option("header", "true").csv("/Users/saicharan/text_classifier/data/Fake_big.csv")
      .withColumn("label", lit(0).cast("double")) // Fake = 0
    val trueDF = spark.read.option("header", "true").csv("/Users/saicharan/text_classifier/data/True_big.csv")
      .withColumn("label", lit(1).cast("double")) // True = 1

    val dataDF = fakeDF.union(trueDF).na.drop()

    // ---------- Split Train / Test ----------
    val Array(trainDF, testDF) = dataDF.randomSplit(Array(0.8, 0.2), seed = 42)

    // ---------- Text Processing Pipeline ----------
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("\\W+") // split by non-word

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(10000)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val nb = new NaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setSmoothing(1.0)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, nb))

    // ---------- Train Model ----------
    val model = pipeline.fit(trainDF)

    // ---------- Predict on Test Data ----------
    val predictions = model.transform(testDF)
    predictions.select("text", "label", "prediction").show(10, truncate = false)

    // ---------- Evaluate Model ----------
    val safePredictions = predictions
      .withColumn("prediction", col("prediction").cast("double"))
      .withColumn("label", col("label").cast("double"))

    val predictionAndLabels = safePredictions
      .select("prediction", "label")
      .rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label")))

    val metrics = new MulticlassMetrics(predictionAndLabels)

    val accuracy = metrics.accuracy
    val precision = metrics.weightedPrecision
    val recall = metrics.weightedRecall
    val f1Score = metrics.weightedFMeasure
    val confusionMatrix = metrics.confusionMatrix

    val timestamp = LocalDateTime.now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))

    println(s"\n========== Evaluation Report ==========")
    println(s"Timestamp: $timestamp")
    println(f"Accuracy     : ${accuracy * 100}%.2f %%")
    println(f"Precision    : ${precision * 100}%.2f %%")
    println(f"Recall       : ${recall * 100}%.2f %%")
    println(f"F1 Score     : ${f1Score * 100}%.2f %%")
    println("\nConfusion Matrix:")
    println(confusionMatrix)
    println("======================================\n")

    // ---------- Interactive Prediction ----------
    val reader = new java.io.BufferedReader(new java.io.InputStreamReader(System.in))
    var continue = true

    while (continue) {
      println("Enter a sentence to classify (or type 'exit' to quit):")
      val sentence = reader.readLine()
      if (sentence == null || sentence.toLowerCase() == "exit") {
        continue = false
      } else {
        val inputDF = Seq((sentence)).toDF("text")
        val prediction = model.transform(inputDF).select("prediction").collect()(0).getDouble(0)
        val labelStr = if (prediction == 1.0) "True" else "Fake"
        println(s"Prediction: $labelStr\n")
      }
    }

    spark.stop()
  }
}

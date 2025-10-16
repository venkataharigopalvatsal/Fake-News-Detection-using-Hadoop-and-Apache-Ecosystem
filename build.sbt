name := "TextClassifier"

version := "0.1"

scalaVersion := "2.13.16"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.0.0",
  "org.apache.spark" %% "spark-sql" % "4.0.0",
  "org.apache.spark" %% "spark-mllib" % "4.0.0"
)

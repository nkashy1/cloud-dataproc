/*
 * Copyright 2017 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.cloud.ml.samples.criteo

import org.apache.spark.sql._
import org.apache.spark.sql.types._

/**
 * Preprocesses a Criteo data set so that it can be used by a TensorFlow classifier.
 * CriteoPreprocessor instances are expected to be called by their `execute` methods.
 *
 * When executed, the CriteoPreprocessor preprocesses the designated input data and exports the
 * results to tfrecords files in a specified output directory.
 *
 * @param mode Train, Evaluate, or Predict
 * @param importer A CriteoImporter through which the raw data set can be imported.
 * @param indexer A CriteoIndexer which will allow us to turn our categorical data into numerical
 *                data.
 * @param exporter A CriteoExporter through which the preprocessed data set can be exported.
 * @param spark An implicit Spark session in which all the preprocessing should take place.
 */
class CriteoPreprocessor(val mode: PreprocessingMode, val importer: CriteoImporter,
                         val indexer: CriteoIndexer, val exporter: CriteoExporter)
                        (implicit val spark: SparkSession) {
  import spark.implicits._

  val features = CriteoFeatures(mode)

  /**
   * Execute the preprocessing job that this preprocessor has been configured for.
   */
  def execute() {
    val cleanedDf = importer.criteoImport

    val withEmbeddedCategoriesDf = indexer(cleanedDf)

    val withTargetFeaturesDf = withEmbeddedCategoriesDf.
      select(features.outputLabels.head, features.outputLabels.tail: _*).
      toDF

    val criteoDf = withTargetFeaturesDf.columns.
      foldLeft(withTargetFeaturesDf)((df, col) =>
        df.withColumn(col, withTargetFeaturesDf(col).cast(IntegerType)))

    exporter.criteoExport(criteoDf)
  }
}

/**
 * Union of the different modes in which preprocessing can be done.
 */
sealed trait PreprocessingMode

case object Train extends PreprocessingMode
case object Predict extends PreprocessingMode
case object Evaluate extends PreprocessingMode

/**
 * Converts string for mode into the appropriate `PreprocessingMode` object.
 */
object PreprocessingMode {
  def apply(specifier: String): Option[PreprocessingMode] = specifier.toLowerCase match {
    case "train" => Some(Train)
    case "predict" => Some(Predict)
    case "evaluate" => Some(Evaluate)
    case _ => None
  }
}

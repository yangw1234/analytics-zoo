/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.zoo.examples.nnframes.finetune

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext

import org.apache.spark.sql.SQLContext

/**
 * Created by yang on 18-8-29.
 */
object NNEstimatorExample {

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator").setMaster("local[1]")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = new SQLContext(sc)

    import com.intel.analytics.bigdl.nn._
    import com.intel.analytics.zoo.pipeline.nnframes.NNEstimator
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val model = Sequential().add(Linear(2, 2))
    val criterion = MSECriterion()
    val estimator = NNEstimator(model, criterion)
      .setLearningRate(0.2)
      .setMaxEpoch(40)
    val data = sc.parallelize(Seq(
      (Array(2.0, 1.0), Array(1.0, 2.0)),
      (Array(1.0, 2.0), Array(2.0, 1.0)),
      (Array(2.0, 1.0), Array(1.0, 2.0)),
      (Array(1.0, 2.0), Array(2.0, 1.0))))
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val data2 = sc.parallelize(Seq(
      (Array(2.0, 1.0), Array(1.0, 2.0)),
      (Array(1.0, 2.0), Array(2.0, 1.0)),
      (Array(2.0, 1.0), Array(1.0, 2.0)),
      (Array(1.0, 2.0), Array(2.0, 1.0))))
    val df2 = sqlContext.createDataFrame(data2).toDF("features", "label")

    val dlModel = estimator.fit(df)

    val prediction1 = dlModel.transform(df)
    val prediction2 = dlModel.transform(df2)


    prediction2.show()
    prediction1.show()

  }
}

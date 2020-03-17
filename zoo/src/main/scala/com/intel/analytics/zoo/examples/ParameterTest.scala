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
package com.intel.analytics.bigdl

import com.intel.analytics.bigdl.parameters.AllReduceParameterV2
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.SparkContext

object ParameterTest {
  LoggerFilter.redirectSparkInfoLogs()


  def main(args: Array[String]): Unit = {

    val parameterSize = args(0).toInt
    val conf = Engine.createSparkConf().setAppName("AllReduceTest")
      .set("spark.task.maxFailures", "1")
//      .set("spark.master", "spark://bdt:7077")
//      .set("spark.executor.cores", "2").set("spark.cores.max", "6")
//    val parameterSize = 1000
    val SparkConf = NNContext.createSparkConf(conf)
    val sc = NNContext.initNNContext(SparkConf)

    val nodeNumber =  EngineRef.getNodeNumber()
    val coreNumer = EngineRef.getCoreNumber()
    val originRdd = sc.parallelize(
      Array.tabulate(nodeNumber * 20)(_ => 0), nodeNumber * 10)
      .mapPartitions(_ => (0 until 20).toIterator)
      .coalesce(nodeNumber).cache()
    val parameter = AllReduceParameterV2.newParameter[Float](nodeNumber, parameterSize)
    originRdd.mapPartitionsWithIndex( { case (index, iter) =>
      val tensor = Tensor[Float](parameterSize)
      Engine.setNodeAndCore(nodeNumber, coreNumer)
      parameter.init(tensor)
      Iterator.single(1)
    }, true).reduce(_ + _)

    val startTime = System.nanoTime()
    var i = 0
    while (i < 100) {
      originRdd.mapPartitionsWithIndex( { case (index, iter) =>
        val tensor = Tensor[Float](parameterSize).fill(index.toFloat)
        parameter.putGradients(tensor)
        Iterator.single(1)
      }, true).reduce(_ + _)

      val result = originRdd.mapPartitionsWithIndex( { case (index, iter) =>
        parameter.aggregateGradientPartition(nodeNumber)
        val part = parameter.gradientPartition
        Iterator.single(part.apply(Array(1)))
      }, true).reduce(_ + _) / nodeNumber
      println(result)
      i += 1
    }
    val endTime = System.nanoTime()
    println(s"single iteration using time ${(endTime - startTime)/100.0/1e9}s")

    sc.stop()
  }
}

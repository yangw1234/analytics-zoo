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


package com.intel.analytics.zoo.examples.tfnet

import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.{Contiguous, Sequential, Transpose}
import com.intel.analytics.bigdl.optim.Top5Accuracy
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.hadoop.io.Text
import org.apache.log4j.{Level, Logger}

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)


  import Options._

  val imageSize = 224

  def main(args: Array[String]) {
    testParser.parse(args, new TestParams()).foreach { param =>

      val sc = NNContext.initNNContext("TFNet Perf")

      // We set partition number to be node*core, actually you can also assign other partitionNum
      val partitionNum = param.partitionNum
      val rawData =
        sc.sequenceFile(param.dataPath, classOf[Text], classOf[Text], partitionNum)
        .map(image => {
          ByteRecord(image._2.copyBytes(), DataSet.SeqFileFolder.readLabel(image._1).toFloat + 1)
        }).coalesce(partitionNum, true)

      val count = rawData.count()

      val transformer = BytesToBGRImg() -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
        HFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) -> BGRImgToSample()
      val evaluationSet = transformer(rawData)

      val model = Sequential[Float]()
      model.add(Transpose(Array((2, 4), (2, 3))))
      model.add(Contiguous())
      model.add(TFNet(param.modelPath))

      println("evaluation start")
      val startTime = System.nanoTime()
      val result = model.evaluate(evaluationSet,
        Array(new Top5Accuracy[Float]), param.batchSize)
      val endTime = System.nanoTime()
      val seconds = (endTime - startTime)/1.0e9
      println(s"evaluation end, using $seconds seconds," +
        s" process $count records, throughput is ${count / seconds}")
      println(s"Configuration is $param")


      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}
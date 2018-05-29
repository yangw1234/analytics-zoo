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

import java.nio.file.Paths

import com.intel.analytics.bigdl.{Module, TestModelBroadcast}
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, SampleToMiniBatch, Transformer}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Contiguous, Sequential, Transpose}
import com.intel.analytics.bigdl.optim.Predictor
import com.intel.analytics.bigdl.optim.Predictor.{predictSamples, splitBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.{ImageMatToTensor, ImageResize, ImageSet, ImageSetToSample}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import scala.reflect.ClassTag


object PerfTest {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(
    image: String = "/tmp/datasets/cat_dog/train_sampled_test",
    outputFolder: String = "data/demo",
    model: String = "/home/yang/applications/faster_rcnn_resnet101_coco_2018_01_28" +
      "/frozen_inference_graph.pb",
    classNamePath: String = "/tmp/models/coco_classname.txt",
    inputNode: String = "ToFloat_3",
    nPartition: Int = 1)

  val parser = new OptionParser[PredictParam]("TFNet Object Detection Example") {
    head("TFNet Object Detection Example")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
    opt[String]('c', "classNamePath")
      .text("where you put the class name file")
      .action((x, c) => c.copy(outputFolder = x))
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
    opt[String]("inputNode")
      .text("inputNode")
      .action((x, c) => c.copy(inputNode = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>

      val sc = NNContext.getNNContext("TFNet Object Detection Example")

      val inputs = Seq(s"${params.inputNode}:0")
      val outputs = Seq("num_detections:0", "detection_boxes:0",
        "detection_scores:0", "detection_classes:0")

      val detector = TFNet(params.model, inputs, outputs)
      val model = Sequential()
      model.add(Transpose(Array((2, 4), (2, 3))))
      model.add(Contiguous())
      model.add(detector)

      val data = ImageSet.read(params.image, sc, minPartitions = params.nPartition)
        .transform(ImageResize(256, 256) -> ImageMatToTensor() -> ImageSetToSample())


      val output = predict(data.toImageFrame(), model)

      // print the first result
      val result = output.toDistributed().rdd.collect()
      println(result)
    }
  }

  private def predict(imageFrame: ImageFrame, model: Module[Float]) = {
    val rdd = imageFrame.asInstanceOf[DistributedImageFrame].rdd
    val modelBroad = TestModelBroadcast[Float]().broadcast(rdd.sparkContext, model.evaluate())
    val partitionNum = rdd.partitions.length
    val toBatchBroad = rdd.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = partitionNum * 1,
      partitionNum = Some(partitionNum),
      featurePaddingParam = None), false)
    val result = rdd.mapPartitions(partition => {
      val copyStart = System.nanoTime()
      val localModel = modelBroad.value()
      val copyEnd = System.nanoTime()
      val cloneTime = (copyEnd - copyStart) / 1.0e6
      logger.info(s"Model Clone time: ${1.0 * (copyEnd - copyStart) / 1e6}ms")
      val copyTransStart = System.nanoTime()
      val localToBatch = toBatchBroad.value._1.cloneTransformer()
      val copyTransEnd = System.nanoTime()
      logger.info(s"Transformer Clone time: ${1.0 * (copyTransEnd - copyTransStart) / 1e6}ms")

      val pipelineStart = System.nanoTime()
      val re = partition.grouped(1).flatMap(imageFeatures => {
        predictImageBatch[Float](localModel, imageFeatures, null, ImageFeature.predict,
          localToBatch, false)
      })
      val arr = re.toArray
      val pipelineEnd = System.nanoTime()
      val singleForwardTime = ((pipelineEnd - pipelineStart) / 1.0e6) / arr.length
      logger.info(s"pipeline calc time: ${1.0 * (pipelineEnd - pipelineStart) / 1e6}ms")
      logger.info(s"single forward time: ${singleForwardTime}")
      logger.info(s"clone time ration ${cloneTime / singleForwardTime}")
      arr.toIterator
    })
    ImageFrame.rdd(result)
  }

  private def predictImageBatch[T: ClassTag](
            localModel: Module[T], imageFeatures: Seq[ImageFeature],
            outputLayer: String, predictKey: String,
            localToBatch: Transformer[Sample[T], MiniBatch[T]],
            shareBuffer: Boolean)(implicit ev: TensorNumeric[T]): Seq[ImageFeature] = {
    val validImageFeatures = imageFeatures.filter(_.isValid)
    val samples = validImageFeatures.map(x => x[Sample[T]](ImageFeature.sample))
    val batchOut = predictSamples(localModel, samples, localToBatch, shareBuffer, outputLayer)
    validImageFeatures.toIterator.zip(batchOut).foreach(tuple => {
      tuple._1(predictKey) = tuple._2
    })
    imageFeatures
  }

  private def predictSamples[T: ClassTag]
  (localModel: Module[T], samples: Seq[Sample[T]],
   localToBatch: Transformer[Sample[T], MiniBatch[T]],
   shareBuffer: Boolean,
   outputLayer: String = null)(implicit ev: TensorNumeric[T]): Iterator[Activity] = {
    val layer = if (outputLayer == null) {
      localModel
    } else {
      val ol = localModel(outputLayer)
      require(ol.isDefined, s"cannot find layer that map name $outputLayer")
      ol.get
    }
    localToBatch(samples.toIterator).flatMap(batch => {
      localModel.forward(batch.getInput())
      splitBatch[T](layer.output, shareBuffer, batch.size())
    })
  }

  private def splitBatch[T: ClassTag](output: Activity, shareBuffer: Boolean, batchSize: Int)
                                            (implicit ev: TensorNumeric[T]): Array[Activity] = {
    val out = if (output.isTensor) {
      val result = if (shareBuffer) output.toTensor[T] else output.toTensor[T].clone()
      if (result.dim() == 1) {
        require(batchSize == 1,
          s"If result dim == 1, the batchSize is required to be 1, while actual is $batchSize")
        Array(result)
      } else {
        result.split(1)
      }
    } else {
      val result = output.toTable
      val first = result[Tensor[T]](1)
      if (first.dim() == 1) {
        require(batchSize == 1,
          s"If result dim == 1, the batchSize is required to be 1, while actual is $batchSize")
        val table = if (shareBuffer) {
          result
        } else {
          val table = T()
          (1 to result.length()).foreach(key => {
            table.insert(result[Tensor[T]](key).clone())
          })
          table
        }
        Array(table)
      } else {
        val batch = first.size(1)
        require(batch == batchSize, s"output batch $batch is not equal to input batch $batchSize")
        val tables = new Array[Table](batch)
        var i = 1
        while (i <= batch) {
          val table = T()
          tables(i - 1) = table
          (1 to result.length()).foreach(key => {
            val split = result[Tensor[T]](key)(i)
            if (shareBuffer) {
              table.insert(split)
            } else {
              table.insert(split.clone())
            }
          })
          i += 1
        }
        tables
      }
    }
    out.asInstanceOf[Array[Activity]]
  }

  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}

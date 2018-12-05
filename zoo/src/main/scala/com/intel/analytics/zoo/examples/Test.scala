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

package com.intel.analytics.zoo.examples

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.log4j.{Level, Logger}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.examples.tfnet.Predict.{PredictParam, parser}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.pipeline.api.net.TFNet.SessionConfig
import scopt.OptionParser


object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class TestParam(baseDir: String = "/home/yang/sources/datasets",
                          batchPerThread: Int = 4, partitions: Int = 112, mode: Int = 0)

  val parser = new OptionParser[TestParam]("TFNet Object Detection Example") {
    head("TFNet Performance Test")
    opt[String]('d', "directory")
      .text("the directory containing model files")
      .action((x, c) => c.copy(baseDir = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchPerThread = x))
    opt[Int]('p', "partitions")
      .text("The number of partitions")
      .action((x, c) => c.copy(partitions = x))
    opt[Int]('m', "mode")
      .text("test mode, 0 for tfnet, 1 for tfnet disable remapper, 2 for analytics-zoo")
      .action((x, c) => c.copy(mode = x))
  }


  def main(args: Array[String]): Unit = {
    val sc = NNContext.initNNContext("TFNet Performance Test")

    parser.parse(args, TestParam()).foreach { params =>
      def testingAnalyticsZoo(): Unit = {
        val models = Array("inception-v1", "mobilenet-v2",
          "mobilenet", "densenet-161", "resnet-50", "vgg-16", "vgg-19")

        val results = for (model <- models) yield {
          val rdd = sc.parallelize(0 until params.partitions, params.partitions)
          val dataset = rdd.flatMap { _ =>
            for (i <- 1 to 100) yield {
              Tensor[Float](3, 224, 224).rand()
            }
          }.map(t => Sample[Float](t))
          val net = Net.loadBigDL[Float](
            s"/opt/work/analytics-zoo-models/analytics-zoo_${model}_imagenet_0.1.0.model")

          val start = System.nanoTime()
          val result = net.predict(dataset, batchPerThread = params.batchPerThread)
          result.map(x => 1).reduce((x, y) => 1)
          val end = System.nanoTime()
          val throughput = 100.0 * 28 * 1e9/ (end - start)
          throughput
        }

        logger.info("analytics-zoo result:")
        models.zip(results).foreach { case (model, result) =>
          logger.info(s"$model throughput is $result records/s")
        }
        logger.info(models.mkString("\t"))
        logger.info(results.mkString("\t"))
      }

      def testiningSlimModels(remapping: Boolean): Unit = {
        val models = Array("tfnet_inception_v1", "tfnet_resnet_v2_101",
          "tfnet_vgg_16", "tfnet_vgg_19")

        val results = for (model <- models) yield {
          val rdd = sc.parallelize(0 until params.partitions, params.partitions)
          val dataset = rdd.flatMap { _ =>
            for (i <- 1 to 100) yield {
              Tensor[Float](224, 224, 3).rand()
            }
          }.map(t => Sample[Float](t))
          val net = TFNet(s"/opt/work/slim-models-tfnet/$model",
            SessionConfig(graphRewriteRemapping = remapping))

          val start = System.nanoTime()
          val result = net.predict(dataset, batchPerThread = params.batchPerThread)
          result.map(x => 1).reduce((x, y) => 1)
          val end = System.nanoTime()
          val throughput = 100.0 * 28 * 1e9/ (end - start)
          throughput
        }

        logger.info(s"slim models result (remapping: $remapping):")
        models.zip(results).foreach { case (model, result) =>
          logger.info(s"$model throughput is $result records/s")
        }
        logger.info(models.mkString("\t"))
        logger.info(results.mkString("\t"))

      }

      def testingKerasModels(remapping: Boolean): Unit = {
        val models = Array("tfnet_DenseNet169", "tfnet_InceptionResNetV2",
          "tfnet_InceptionV3", "tfnet_MobileNet", "tfnet_MobileNetV2", "tfnet_NASNetLarge",
          "tfnet_NASNetMobile", "tfnet_ResNet50", "tfnet_VGG16", "tfnet_VGG19", "tfnet_Xception")

        val results = for (model <- models) yield {
          val rdd = sc.parallelize(0 until params.partitions, params.partitions)
          val dataset = rdd.flatMap { _ =>
            for (i <- 1 to 100) yield {
              Tensor[Float](224, 224, 3).rand()
            }
          }.map(t => Sample[Float](t))
          val net = TFNet(s"/opt/work/keras-models-tfnet/$model",
            SessionConfig(graphRewriteRemapping = remapping))

          val start = System.nanoTime()
          val result = net.predict(dataset, batchPerThread = params.batchPerThread)
          result.map(x => 1).reduce((x, y) => 1)
          val end = System.nanoTime()
          val throughput = 100.0 * 28 * 1e9/ (end - start)
          throughput
        }

        logger.info(s"keras models result (remapping: $remapping):")
        models.zip(results).foreach { case (model, result) =>
          logger.info(s"$model throughput is $result records/s")
        }
        logger.info(models.mkString("\t"))
        logger.info(results.mkString("\t"))
      }

      def testingObjectDetectionModels(remapping: Boolean): Unit = {
        val models = Array("ssd_mobilenet_v1_coco_2018_01_28", "ssd_mobilenet_v2_coco_2018_03_29/",
          "ssd_inception_v2_coco_2018_01_28", "faster_rcnn_inception_v2_coco_2018_01_28",
          "faster_rcnn_resnet50_coco_2018_01_28", "faster_rcnn_resnet101_coco_2018_01_28")

        val results = for (model <- models) yield {
          val rdd = sc.parallelize(0 until params.partitions, params.partitions)
          val dataset = rdd.flatMap { _ =>
            for (i <- 1 to 100) yield {
              Tensor[Float](600, 600, 3).rand()
            }
          }.map(t => Sample[Float](t))
          val net = TFNet(s"/opt/work/tensorflow-object-detection/$model/frozen_inference_graph.pb",
            inputNames = Array("image_tensor:0"),
            outputNames = Array("num_detections:0", "detection_boxes:0",
              "detection_scores:0", "detection_classes:0"),
            SessionConfig(graphRewriteRemapping = remapping))

          val start = System.nanoTime()
          val result = net.predict(dataset, batchPerThread = params.batchPerThread)
          result.map(x => 1).reduce((x, y) => 1)
          val end = System.nanoTime()
          val throughput = 100.0 * 28 * 1e9/ (end - start)
          throughput
        }

        logger.info(s"object detection models result (remapping: $remapping):")
        models.zip(results).foreach { case (model, result) =>
          logger.info(s"$model throughput is $result records/s")
        }

        logger.info(models.mkString("\t"))
        logger.info(results.mkString("\t"))
      }

      params.mode match {
        case 0 =>
          testingKerasModels(true)
          testiningSlimModels(true)
          testingObjectDetectionModels(true)

        case 1 =>
          testingKerasModels(false)
          testiningSlimModels(false)
          testingObjectDetectionModels(false)

        case 2 =>
          testingAnalyticsZoo()
      }

    }


  }
}

package com.intel.analytics.zoo.examples.nnframes.finetune

import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.utils.{LoggerFilter, Shape}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNImageReader, NNImageTransformer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser


object Utils {

  case class TrainParams(
                          modelPath: String = "/tmp/bigdl_inception-v1_imagenet_0.4.0.model",
                          dataPath: String = "/tmp/train_sampled",
                          batchSize: Int = 32,
                          nEpochs: Int = 2)

  val trainParser = new OptionParser[TrainParams]("BigDL ptbModel Train Example") {
    opt[String]('m', "modelPath")
      .text("pretrained model path")
      .action((x, c) => c.copy(dataPath = x))
    opt[String]('d', "dataPath")
      .text("training data path")
      .action((x, c) => c.copy(dataPath = x))
    opt[Int]('b', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }
}

object TransferLearning {

  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {
    Utils.trainParser.parse(args, Utils.TrainParams()).map(param => {

      val conf = new SparkConf()
        .setAppName("Transfer Learning Example")

      val sc = NNContext.getNNContext(conf)

      val model = getTransferLearningModel(param.modelPath)

      val (trainDf, valDf) = getImageData(param.dataPath, sc)

      val featureTransformer = createTransformer()

      val valTransformed = featureTransformer.transform(valDf)
      val classifier = new NNClassifier(model, CrossEntropyCriterion[Float](), Array(3, 224, 224))
        .setFeaturesCol("output")
        .setLearningRate(0.003)
        .setBatchSize(param.batchSize)
        .setMaxEpoch(param.nEpochs)
        .setValidation(Trigger.everyEpoch, valTransformed, Array(new Top1Accuracy()), param.batchSize)


      val pipeline = new Pipeline().setStages(Array(featureTransformer, classifier))

      val pipelineModel = pipeline.fit(trainDf)

      val predictions = pipelineModel.transform(valDf)

      predictions.select(col("image"), col("label"), col("prediction"))
        .sample(withReplacement = true, 0.1)
        .show(false)

      sc.stop()
    })
  }

  private def getTransferLearningModel(preTrainedPath: String) = {
    // you can use Net.loadBigDL[Float](preTrainedPath).saveGraphTopology(somePath)
    // and use tensorboard to visualize the model topology and decide
    val inception = Net
      .loadBigDL[Float](preTrainedPath)
      .newGraph(output = "pool5/drop_7x7_s1") // remove layers after pool5/drop_7x7_s1

    inception.freezeUpTo("pool4/3x3_s2") // freeze layer pool4/3x3_s2 and the layers before it

    // add a new classifer
    val input = Input[Float](inputShape = Shape(3, 224, 224))
    val feature = new KerasLayerWrapper[Float](inception).inputs(input)
    val flattern = Flatten[Float]().inputs(feature)
    val logits = Dense[Float](2).inputs(flattern)

    Model(input, logits)
  }

  private def createTransformer() = {
    new NNImageTransformer(
      Resize(256, 256) ->
        CenterCrop(224, 224) ->
        ChannelNormalize(123, 117, 104) ->
        MatToTensor() ->
        ImageFrameToSample())
  }

  private def getImageData(dataPath: String, sc: SparkContext) = {
    val createLabel = udf { row: Row => if (row.getString(0).contains("cat")) 1.0 else 2.0 }
    val imagesDF = NNImageReader.readImages(dataPath, sc)
      .withColumn("label", createLabel(col("image")))
    val Array(validationDF, trainingDF) = imagesDF
      .randomSplit(Array(0.20, 0.80), seed = 1L)
    (trainingDF, validationDF)
  }

}

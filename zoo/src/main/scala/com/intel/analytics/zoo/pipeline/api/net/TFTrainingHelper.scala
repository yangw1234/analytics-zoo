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

package com.intel.analytics.zoo.pipeline.api.net

import java.nio._

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import org.apache.spark.rdd.RDD
import org.tensorflow.{Tensor => TTensor}

import scala.collection.Iterator
import scala.io.Source
import scala.reflect.io.Path

private[zoo] class TFTrainingHelper(tfnet: TFNet)
  extends AbstractModule[Activity, Activity, Float] {

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    tfnet.parameters()
  }
  private val weights = tfnet.parameters()._1

  private val gradWeights = tfnet.parameters()._2


  private def tf2bigdl(t: TTensor[_], output: Tensor[Float]) = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val buffer = FloatBuffer.wrap(
      output.storage().array(),
      output.storageOffset() - 1,
      shape.product)
    t.writeTo(buffer)
  }

  override def updateOutput(input: Activity): Activity = {
    tfnet.evaluate()
    tfnet.assignWeights()
    val feeds = T()
    if (input.isTensor) {
      feeds.insert(input)
    } else {
      var i = 0
      while (i < input.toTable.length()) {
        feeds.insert(input.toTable(i + 1))
        i += 1
      }

    }

    val fetches = tfnet.forward(feeds).toTable.toSeq[Tensor[Float]].toArray

    var i = 0
    val len = weights.length
    while (i < len) {
      gradWeights(i).resizeAs(weights(i)).add(fetches(i))
      i += 1
    }

    val offset = len
    val allLength = fetches.length

    output = if (allLength == len + 1) {
      fetches(offset)
    } else {
      val result = T()
      var i = offset
      while (i < allLength) {
        result.insert(fetches(i))
        i += 1
      }
      result
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }
}

object TFTrainingHelper {

  def apply(modelPath: String): TFTrainingHelper = {

    val tfnet = TFNet(modelPath)

    new TFTrainingHelper(tfnet)
  }
}


class IdentityCriterion extends AbstractCriterion[Activity, Activity, Float]() {

  override def updateOutput(input: Activity, target: Activity): Float = {
    if (input.isTensor) {
      input.toTensor[Float].value()
    } else {
      val table = input.toTable
      table[Tensor[Float]](table.length()).value()
    }
  }
  override def updateGradInput(input: Activity, target: Activity): Activity = {
    gradInput
  }
}

class TFValidationMethod(val valMethod: ValidationMethod[Float],
                         outputLength: Int,
                         targetLength: Int) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., outputs..., labels..., loss]
    val outputT = output.toTable

    if (valMethod.isInstanceOf[Loss[Float]]) {
      val loss = outputT[Tensor[Float]](outputT.length()).value()
      return new LossResult(loss, 1)
    }
    val outputActivity: Activity = if (outputLength == 1) {
      outputT[Tensor[Float]](outputT.length() - outputLength - targetLength)
    } else {
      var i = outputT.length() - outputLength - targetLength
      val outputs = T()
      while (i < outputLength - targetLength) {
        outputs.insert(outputT(i))
          i += 1
      }
      outputs
    }

    val to1basedLabel = !valMethod.isInstanceOf[Accuracy[Float]] &&
      valMethod.isInstanceOf[Top1Accuracy[Float]] ||
        valMethod.isInstanceOf[Top5Accuracy[Float]] ||
        valMethod.isInstanceOf[TreeNNAccuracy[Float]]
    val targetActivity = if (targetLength == 1) {
      val t = outputT[Tensor[Float]](outputT.length() - targetLength)
      if (to1basedLabel) t.add(1.0f)
      t
    } else {
      var i = outputT.length() - targetLength
      val targets = T()
      while (i < outputLength) {
        val t = outputT[Tensor[Float]](i)
        if (to1basedLabel) t.add(1.0f)
        targets.insert(t)
        i += 1
      }
      targets
    }

    valMethod.apply(outputActivity, targetActivity)
  }

  override protected def format(): String = {
    valMethod.toString()
  }
}

class MergeFeatureLabel() extends Transformer[Sample[Float], Sample[Float]] {
  override def apply(prev: Iterator[Sample[Float]]): Iterator[Sample[Float]] = {
    new Iterator[Sample[Float]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): Sample[Float] = {
        val oldSample = prev.next()
        val newSize = oldSample.getFeatureSize() ++ oldSample.getLabelSize()
        Sample(oldSample.getData(), newSize, null)
      }
    }
  }
}

case class TrainMeta(inputNames: Array[String], outputNames: Array[String],
                     variables: Array[String], gradVariables: Array[String])


class TFOptimizer(modelPath: String,
                  optimMethod: OptimMethod[Float],
                  x: RDD[Sample[Float]],
                  batchSize: Int = 32) {
  private val trainer: TFTrainingHelper = TFTrainingHelper(modelPath)
  private val optimizer: Optimizer[Float, MiniBatch[Float]] = {
    val optimizer = Optimizer[Float](trainer, x, new IdentityCriterion(), batchSize)

    optimizer.setOptimMethod(optimMethod)
    optimizer
  }

  def optimize(endTrigger: Trigger = Trigger.maxEpoch(1)): Array[Tensor[Float]] = {
    optimizer.setEndWhen(endTrigger)
    optimizer.optimize()
    trainer.parameters()._1
  }
}



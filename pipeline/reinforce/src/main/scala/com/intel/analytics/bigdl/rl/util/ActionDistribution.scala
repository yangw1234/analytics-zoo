/*
  * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.rl.util

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

abstract class ActionDistribution[T: ClassTag]()(implicit ev: TensorNumeric[T]) {

  def logp(actions: Tensor[T]): Tensor[T]

  def kl(other: ActionDistribution[T]): Tensor[T]

  def negativeEntropy(): Tensor[T]

}


class Categorical[T: ClassTag](val probabilities: Tensor[T])
                              (implicit ev: TensorNumeric[T]) extends ActionDistribution[T] {

  private val batchSize = probabilities.size(1)
  private val logpResultBuffer: Tensor[T] = Tensor[T]()
  private val klResultBuffer: Tensor[T] = Tensor[T]()
  private val entropyBuffer: Tensor[T] = {
    val result = Tensor[T](Array(batchSize))
    val buffer = Tensor[T]()
    buffer.resizeAs(probabilities)
      .copy(probabilities).log().cmul(probabilities)
    result.sum(buffer, 2)
  }

  override def logp(actions: Tensor[T]): Tensor[T] = {
    logpResultBuffer.resize(batchSize)
    var i = 0
    while (i < actions.size(1)) {
      val prob = probabilities.valueAt(i + 1, ev.toType[Int](actions.valueAt(i + 1)) + 1)
      logpResultBuffer.setValue(i + 1, prob)
      i += 1
    }
    logpResultBuffer
  }

  override def kl(other: ActionDistribution[T]): Tensor[T] = {
    val otherCategorical = other.asInstanceOf[Categorical[T]]
    val buffer = Tensor[T]()
    buffer.resizeAs(probabilities).copy(probabilities)
      .div(otherCategorical.probabilities).log().cmul(probabilities)
    klResultBuffer.sum(buffer, 2)
    klResultBuffer
  }

  override def negativeEntropy() = {
    entropyBuffer
  }
}


class DiagGaussian[T: ClassTag](val mean: Tensor[T], val logStd: Tensor[T])
                               (implicit ev: TensorNumeric[T]) extends ActionDistribution[T] {
  private val const1 = 0.5 * math.log(2.0 * math.Pi)
  private val const2 = .5 * math.log(2.0 * math.Pi * math.E)
  private val batchSize = mean.size(1)
  private val logpResultBuffer: Tensor[T] = Tensor[T]()
  private val klResultBuffer: Tensor[T] = Tensor[T]()
  private val std = {
    val stdBuffer = Tensor[T]()
    stdBuffer.resizeAs(logStd).copy(logStd).exp()
    stdBuffer
  }

  lazy val variances = {
    val t = Tensor[T]()
    t.resizeAs(std).copy(std).square()
  }
  private val entropyBuffer = {
    val buffer = Tensor[T]()
    val entropy = Tensor[T]().resize(batchSize)
    buffer.resizeAs(logStd)
      .copy(logStd).add(ev.fromType(const2))
    entropy.sum(buffer, 2)
    entropy.mul(ev.fromType(-1))
  }


  override def logp(actions: Tensor[T]): Tensor[T] = {
    val buffer = Tensor[T]()
    val logp = buffer.resizeAs(actions).copy(actions).sub(mean).cdiv(std).square().sum(2).mul(ev.fromType(-0.5))
    logp.sub(ev.fromType(const1 * mean.size(2))).sub(logStd.sum(2))
    logp
  }

  override def kl(other: ActionDistribution[T]): Tensor[T] = {

    val otherDiagGuassian = other.asInstanceOf[DiagGaussian[T]]
    val firstTerm = Tensor[T]()
    firstTerm.resizeAs(this.logStd).copy(this.logStd).sub(otherDiagGuassian.logStd)

    val thisStdSquare = Tensor[T]().resizeAs(this.std).copy(this.std).square()
    val otherStdSquare = Tensor[T]().resizeAs(otherDiagGuassian.std).copy(otherDiagGuassian.std).square()
    val meanDiffSquare = Tensor[T]().resizeAs(this.mean).copy(this.mean).sub(otherDiagGuassian.mean).square()

    val secondTerm = Tensor[T]()
    secondTerm.resizeAs(thisStdSquare).copy(thisStdSquare).add(meanDiffSquare).div(otherStdSquare).div(ev.fromType(2.0))

    firstTerm.sub(secondTerm).sub(ev.fromType(0.5))

    klResultBuffer.sum(firstTerm, 2)

    klResultBuffer
  }

  override def negativeEntropy() = {
    entropyBuffer
  }

}

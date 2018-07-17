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

package com.intel.analytics.zoo.pipeline.api.torch

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class CPPOCriterion[T: ClassTag](
                                           epsilon: Double = 0.3,
                                           entropyCoeff: Double = 0.0,
                                           klTarget: Double = 0.01,
                                           initBeta: Double = 0.0
                                         )
                                         (implicit ev: TensorNumeric[T])
  extends AbstractCriterion[Table, Tensor[T], T] {

  private val ratio: Tensor[T] = Tensor[T]()
  private val clippedRatio: Tensor[T] = Tensor[T]()
  private val buffer: Tensor[T] = Tensor[T]()
  private val gradMean: Tensor[T] = Tensor[T]()
  private val gradLogStd: Tensor[T] = Tensor[T]()
  private val distributionBuffer: Tensor[T] = Tensor[T]()
  private val surr1: Tensor[T] = Tensor[T]()
  private val surr2: Tensor[T] = Tensor[T]()
  private var kld: T = ev.zero
  private var beta: Double = initBeta
  private val mask: Tensor[T] = Tensor[T]()

  private var currentDist: DiagGaussian[T] = null
  private var oldDist: DiagGaussian[T] = null
  private var currentLogp: Tensor[T] = null
  private var oldLogp: Tensor[T] = null


  override def updateOutput(input: Table, target: Tensor[T]): T = {

    val mean = input[Tensor[T]](1)
    val logStd = input[Tensor[T]](2)

    val batchSize = mean.size(1)

    val actionSize = mean.size(2)
    val actions = target.narrow(2, 1, actionSize).clone() // batchSize x actionSize
    val advantage = target.select(2, actionSize + 1).clone() // batchSize x 1
    val oldMean = target.narrow(2, actionSize + 2, actionSize).clone() // batchSize X actionSize
    val oldLogStd = target.narrow(2, 2 * actionSize + 2, actionSize).clone() // batchSize X actionSize

    currentDist = new DiagGaussian[T](mean, logStd)
    oldDist = new DiagGaussian[T](oldMean, oldLogStd)

    currentLogp = currentDist.logp(actions)
    oldLogp = oldDist.logp(actions)

    ratio.resizeAs(advantage).copy(currentLogp).sub(oldLogp).exp()

    clippedRatio.resizeAs(ratio).copy(ratio)
      .clamp(1.0f - epsilon.toFloat, 1.0f + epsilon.toFloat)

    surr1.resizeAs(ratio).copy(ratio).cmul(advantage)

    surr2.resizeAs(clippedRatio).copy(clippedRatio).cmul(advantage)

    buffer.resizeAs(surr1).cmin(surr1, surr2)

    output = ev.negative(buffer.mean())

    if (entropyCoeff != 0.0) {
      output = ev.plus(output,
        ev.times(currentDist.negativeEntropy().mean(), ev.fromType(entropyCoeff)))
    }

    if (initBeta != 0.0) {
      kld = oldDist.kl(currentDist).mean()
      output = ev.plus(output, ev.times(kld, ev.fromType(beta)))
    }

    output
  }

  override def updateGradInput(input: Table, target: Tensor[T]): Table = {
    mask.resizeAs(surr1).le(surr1, surr2)

    val mean = input[Tensor[T]](1)
    val logStd = input[Tensor[T]](2)

    val batchSize = mean.size(1)

    val actionSize = mean.size(2)
    val actions = target.narrow(2, 1, actionSize).clone() // batchSize x actionSize
    val advantage = target.select(2, actionSize + 1).clone() // batchSize x 1
    val oldMean = target.narrow(2, actionSize + 2, actionSize).clone() // batchSize X actionSize
    val oldLogStd = target.narrow(2, 2 * actionSize + 2, actionSize).clone() // batchSize X actionSize


    gradMean.resizeAs(actions).copy(actions).sub(mean)
      .div(currentDist.variances).cmul(ratio.view(batchSize, 1))
      .cmul(advantage.view(batchSize, 1))
      .cmul(mask.view(batchSize, 1))
      .mul(ev.fromType(-1.0/batchSize))

    gradLogStd.resizeAs(actions).copy(actions).sub(mean).square()
      .div(currentDist.variances).sub(ev.one)
      .cmul(ratio.view(batchSize, 1))
      .cmul(advantage.view(batchSize, 1))
      .cmul(mask.view(batchSize, 1))
      .mul(ev.fromType(-1.0/batchSize))

    if (entropyCoeff != 0.0) {
      gradLogStd.add(ev.fromType(-1.0 * entropyCoeff /batchSize), logStd)
    }

    if (initBeta != 0.0) {

      val gradKL = Tensor[T]().resizeAs(mean).copy(mean).sub(oldMean).div(oldDist.variances)

      gradMean.add(ev.fromType(beta / batchSize), gradKL)


      if (ev.isGreater(kld, ev.fromType(klTarget * 1.5))) {
        beta = beta * 2.0
      } else if (ev.isGreater(ev.fromType(klTarget / 1.5), kld)) {
        beta = beta / 2.0
      }
    }

    gradInput(1) = gradMean
    gradInput(2) = gradLogStd

    gradInput
  }
}

object CPPOCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
           epsilon: Double = 0.3,
           entropyCoeff: Double = 0.0,
           klTarget: Double = 0.01,
           initBeta: Double = 0.0)(implicit ev: TensorNumeric[T]): CPPOCriterion[T] = {
    new CPPOCriterion[T](epsilon, entropyCoeff, klTarget, initBeta)
  }
}


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

  override def negativeEntropy(): Tensor[T] = {
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
    val logp = buffer.resizeAs(actions).copy(actions)
      .sub(mean).cdiv(std).square().sum(2).mul(ev.fromType(-0.5))
    logp.sub(ev.fromType(const1 * mean.size(2))).sub(logStd.sum(2))
    logp
  }

  override def kl(other: ActionDistribution[T]): Tensor[T] = {

    val otherDiagGuassian = other.asInstanceOf[DiagGaussian[T]]
    val firstTerm = Tensor[T]()
    firstTerm.resizeAs(this.logStd).copy(this.logStd).sub(otherDiagGuassian.logStd)

    val thisStdSquare = Tensor[T]().resizeAs(this.std).copy(this.std).square()
    val otherStdSquare = Tensor[T]().resizeAs(otherDiagGuassian.std)
      .copy(otherDiagGuassian.std).square()
    val meanDiffSquare = Tensor[T]().resizeAs(this.mean).copy(this.mean)
      .sub(otherDiagGuassian.mean).square()

    val secondTerm = Tensor[T]()
    secondTerm.resizeAs(thisStdSquare).copy(thisStdSquare)
      .add(meanDiffSquare).div(otherStdSquare).div(ev.fromType(2.0))

    firstTerm.sub(secondTerm).sub(ev.fromType(0.5))

    klResultBuffer.sum(firstTerm, 2)

    klResultBuffer
  }

  override def negativeEntropy(): Tensor[T] = {
    entropyBuffer
  }

}

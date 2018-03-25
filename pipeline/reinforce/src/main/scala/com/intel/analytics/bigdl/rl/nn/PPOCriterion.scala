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
package com.intel.analytics.bigdl.rl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, TensorCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class PPOCriterion[T: ClassTag](
                                 epsilon: Double = 0.3,
                                 entropyCoeff: Double = 0.0,
                                 klTarget: Double = 0.01,
                                 initBeta: Double = 0.0
                               )
                               (implicit ev: TensorNumeric[T])
  extends AbstractCriterion[Tensor[T], Table, T] {
  private val ratio: Tensor[T] = Tensor[T]()
  private val clippedRatio: Tensor[T] = Tensor[T]()
  private val buffer: Tensor[T] = Tensor[T]()
  private val mask: Tensor[T] = Tensor[T]()
  private val distributionBuffer = Tensor[T]()
  private var beta: Double = initBeta
  private var kld: T = ev.zero

  private var surr1: Tensor[T] = null
  private var surr2: Tensor[T] = null

  private def sumOfNegativeEntropy(dist: Tensor[T]): T = {
    distributionBuffer.resizeAs(dist)
      .copy(dist).log().cmul(dist).sum()
  }

  private def meanOfKLD(p: Tensor[T], q: Tensor[T]): T = {
    distributionBuffer.resizeAs(p).copy(p).div(q).log().cmul(p).mean()
  }

  override def updateOutput(input: Tensor[T], target: Table): T = {
    val batchSize = input.size(1)

    val action = target[Tensor[T]](1).view(batchSize)
    val advantage = target[Tensor[T]](2).view(batchSize)
    val preProbs = target[Tensor[T]](3)

    ratio.resizeAs(advantage)

    var i = 0
    while (i < action.size(1)) {
      val currProb = input.valueAt(i + 1, ev.toType[Int](action.valueAt(i + 1)) + 1)
      val preProb = preProbs.valueAt(i + 1, ev.toType[Int](action.valueAt(i + 1)) + 1)
      ratio.setValue(i + 1, ev.divide(currProb, preProb))
      i += 1
    }

    clippedRatio.resizeAs(ratio).copy(ratio)
      .clamp(1.0f - epsilon.toFloat, 1.0f + epsilon.toFloat)

    surr1 = if (advantage.dim() == 2) {
      ratio.cmul(advantage).sum(2)
    } else {
      ratio.cmul(advantage)
    }
    surr2 = if (advantage.dim() == 2) {
      clippedRatio.cmul(advantage).sum(2)
    } else {
      clippedRatio.cmul(advantage)
    }

    buffer.resizeAs(surr1).cmin(surr1, surr2)

    output = ev.negative(buffer.mean())

    if (entropyCoeff != 0.0) {
      output = ev.plus(output, ev.divide(sumOfNegativeEntropy(input), ev.fromType(batchSize)))
    }

    if (initBeta != 0.0) {
      kld = meanOfKLD(preProbs, input)
      output = ev.plus(output, ev.times(kld, ev.fromType(beta)))
    }

    output
  }

  override def updateGradInput(input: Tensor[T], target: Table): Tensor[T] = {
    mask.resizeAs(surr1).le(surr1, surr2)

    val batchSize = input.size(1)

    val action = target[Tensor[T]](1).view(batchSize)
    val advantage = target[Tensor[T]](2).view(batchSize)
    val preProbs = target[Tensor[T]](3)

    buffer.resizeAs(advantage)
    var i = 0
    while (i < action.size(1)) {
      val preProb = preProbs.valueAt(i + 1, ev.toType[Int](action.valueAt(i + 1)) + 1)
      buffer.setValue(i + 1, ev.divide(ev.one, preProb))
      i += 1
    }

    buffer.cmul(advantage).cmul(mask)
      .mul(ev.fromType(-1.0 / batchSize))

    gradInput.resizeAs(input).zero()

    i = 0
    while (i < action.size(1)) {
      gradInput.setValue(i + 1, ev.toType[Int](action.valueAt(i + 1)) + 1, buffer.valueAt(i + 1))
      i += 1
    }

    if (entropyCoeff != 0.0) {
      distributionBuffer.resizeAs(input).copy(input).log().add(ev.one)
      gradInput.add(distributionBuffer)
    }

    if (initBeta != 0.0) {
      distributionBuffer.resizeAs(input).copy(preProbs).cdiv(input).mul(ev.fromType(-1))
      gradInput.add(distributionBuffer)

      if (ev.isGreater(kld, ev.fromType(klTarget * 1.5))) {
        beta = beta * 2.0
      } else if (ev.isGreater(ev.fromType(klTarget / 1.5), kld)) {
        beta = beta / 2.0
      }
    }

    gradInput
  }
}

object PPOCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      epsilon: Double = 0.3,
                                                      entropyCoeff: Double = 0.0,
                                                      klTarget: Double = 0.01,
                                                      initBeta: Double = 0.0
                                                    )
                                                    (implicit ev: TensorNumeric[T]): PPOCriterion[T] = {
    new PPOCriterion(epsilon, entropyCoeff, klTarget, initBeta)
  }
}

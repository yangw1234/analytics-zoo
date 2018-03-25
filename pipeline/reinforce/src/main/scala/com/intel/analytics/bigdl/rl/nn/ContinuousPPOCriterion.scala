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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.rl.util.DiagGaussian
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class ContinuousPPOCriterion[T: ClassTag](
                                 epsilon: Double = 0.3,
                                 entropyCoeff: Double = 0.0,
                                 klTarget: Double = 0.01,
                                 initBeta: Double = 0.0
                               )
                               (implicit ev: TensorNumeric[T])
  extends AbstractCriterion[Table, Table, T] {

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


  override def updateOutput(input: Table, target: Table): T = {

    val mean = input[Tensor[T]](1)
    val logStd = input[Tensor[T]](2)

    val batchSize = mean.size(1)

    val actions = target[Tensor[T]](1)
    val advantage = target[Tensor[T]](2).view(batchSize)
    val oldMean = target[Tensor[T]](3)
    val oldLogStd = target[Tensor[T]](4)

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

  override def updateGradInput(input: Table, target: Table): Table = {
    mask.resizeAs(surr1).le(surr1, surr2)

    val mean = input[Tensor[T]](1)
    val logStd = input[Tensor[T]](2)

    val batchSize = mean.size(1)

    val actions = target[Tensor[T]](1)
    val advantage = target[Tensor[T]](2).view(batchSize)
    val oldMean = target[Tensor[T]](3)
    val oldLogStd = target[Tensor[T]](4)


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

object ContinuousPPOCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      epsilon: Double = 0.3,
                                                      entropyCoeff: Double = 0.0,
                                                      klTarget: Double = 0.01,
                                                      initBeta: Double = 0.0
                                                    )
                                                    (implicit ev: TensorNumeric[T]): ContinuousPPOCriterion[T] = {
    new ContinuousPPOCriterion[T](epsilon, entropyCoeff, klTarget, initBeta)
  }
}
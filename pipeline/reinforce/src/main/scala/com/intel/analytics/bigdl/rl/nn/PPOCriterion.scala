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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, TensorCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class PPOCriterion[T: ClassTag](
                                epsilon: Float = 0.3f)
                              (implicit ev: TensorNumeric[T])
  extends AbstractCriterion[Tensor[T], Table, T] {
  private val ratio: Tensor[T] = Tensor[T]()
  private val clippedRatio: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Table): T = {
    val advantage = target[Tensor[T]](1)
    val preProb = target[Tensor[T]](2)

    ratio.resizeAs(input)
    ratio.copy(preProb).div(input)

    clippedRatio.resizeAs(ratio).copy(ratio)
      .clamp(1.0f - epsilon, 1.0f + epsilon)

    val surr1 = ratio.cmul(advantage).sum()
    val surr2 = clippedRatio.cmul(advantage).sum()

    output = ev.min(surr1, surr2)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Table): Tensor[T] = {

    gradInput
  }
}

object PGCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      sizeAverage: Boolean = false)
                                                    (implicit ev: TensorNumeric[T]): PGCriterion[T] = {
    new PGCriterion()
  }
}

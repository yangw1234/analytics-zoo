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

package com.intel.analytics.bigdl.rl.python.api


import com.intel.analytics.bigdl.rl.nn.{DiscretePPOCriterion, _}
import com.intel.analytics.bigdl.python.api.{PythonBigDL, PythonBigDLKeras}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._

import scala.reflect.ClassTag
import org.apache.log4j.Logger


object RLPythonBigDL {
  val logger = Logger.getLogger("com.intel.analytics.bigdl.rl.python.api.RLPythonBigDL")

  def ofFloat(): RLPythonBigDL[Float] = new RLPythonBigDL[Float]()

  def ofDouble(): RLPythonBigDL[Double] = new RLPythonBigDL[Double]()

}


class RLPythonBigDL[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }

  def createRLPythonBigDL(): RLPythonBigDL[T] = {
    new RLPythonBigDL[T]()
  }

  def createDiscretePPOCriterion(epsilon: Double,
                         entropyCoeff: Double,
                         klTarget: Double,
                         initBeta: Double
                        ): DiscretePPOCriterion[T] = {
    DiscretePPOCriterion[T](epsilon, entropyCoeff, klTarget, initBeta)
  }

  def createContinuousPPOCriterion(epsilon: Double,
                                   entropyCoeff: Double,
                                   klTarget: Double,
                                   initBeta: Double): ContinuousPPOCriterion[T] = {
    ContinuousPPOCriterion[T](epsilon, entropyCoeff, klTarget, initBeta)
  }

}

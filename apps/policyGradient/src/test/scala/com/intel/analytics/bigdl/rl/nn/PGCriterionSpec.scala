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
 *
 */

package com.intel.analytics.bigdl.rl.nn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.scalatest.{FlatSpec, Matchers}

class PGCriterionSpec extends FlatSpec with Matchers {

  "PGCriterion" should "work correctly in batch mode" in {
    val inputN = 5
    val seed = 100
    //RNG.setSeed(seed)
    val input = Tensor(3,2).fill(0.5f)
    val target = Tensor(3,2)
    target(Array(1, 1)) = 1f //action
    target(Array(1, 2)) = 0.2f //reward
    target(Array(2, 1)) = 1f //action
    target(Array(2, 2)) = -0.5f //reward
    target(Array(3, 1)) = 2f //action
    target(Array(3, 2)) = 0.6f //reward

    val cr = PGCriterion(false,1.2,0.8)
    val out = cr.forward(input,target)
    println(s"out = $out")
    val gradIn = cr.backward(input, target)
    println(s"gradIn = $gradIn")
  }

  "PGCriterion" should "work correctly in non-batch mode" in {
    val inputN = 5
    val seed = 100
    //RNG.setSeed(seed)
    val input = Tensor(5).fill(0.2f)
    val target = Tensor(2)
    target(Array(1)) = 3f //action
    target(Array(2)) = 10f //reward

    val cr = PGCriterion(true,1.2,0.8)
    val out = cr.forward(input,target)
    println(s"out = $out")
    val gradIn = cr.backward(input, target)
    println(s"gradIn = $gradIn")
  }

}

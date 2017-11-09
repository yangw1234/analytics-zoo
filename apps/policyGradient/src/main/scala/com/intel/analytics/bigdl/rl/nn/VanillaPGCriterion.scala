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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.Engine

/**
 * VanillaPGCriterion
 * forward returns 0
 * for backward, target contains 2 values, action and reward. 
 * gradient = (-1)*(action-prob)*reward
 * use together with sigmoid
 *
 * @param sizeAverage size average of batch
 * @param ev numeric operator
 * @tparam T numeric type
 */

class VanillaPGCriterion[@specialized(Float, Double) T: ClassTag]
(sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    //forward is trival here  
    output = ev.zero
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {

    require(input.dim() == 1 || input.dim() == 2,
      "VanillaPGCriterion: " +
              ErrorInfo.constrainInputAsVectorOrBatch +
                      s"input dim ${input.dim()}")

    gradInput.resizeAs(input)
    gradInput.zero()

    //if a single value
    if (input.dim() == 1){
      val targetSize = target.size()
      target.squeeze()
      require(target.dim()==1,s"target dimension should be 1, but is: ${target.dim()}")
      gradInput.setValue(1, 
          ev.times(
            ev.minus(input.valueAt(1),target.valueAt(1)), 
            target.valueAt(2)))
    }
    //if a minibatch 
    else if (input.dim() == 2) {
      //TODO, need to support table as target or more than 1 dim.  
      require(target.dim()==2,s"target dimension should be 2 (with batch dimension, but target dimension is: ${target.dim()}")
      //val batchSize = input.size(1)
      require(target.nElement() == 2*input.size(1),s"target should contain 2*inputsize of elements, but is:${target.nElement()}")
      
      //formula is (-1)*(action-prob)*reward
      val action = target.select(2, 1).contiguous()
      val reward = target.select(2, 2).contiguous()                    
      gradInput.add(input, ev.negative(ev.one), action)
      gradInput.cmul(reward) 

      if (sizeAverage) {
        gradInput.div(ev.fromType[Int](gradInput.nElement()))
      }
    }
  
    gradInput
    
  }

}

object VanillaPGCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : VanillaPGCriterion[T] = {
    new VanillaPGCriterion[T](sizeAverage)
  }
}

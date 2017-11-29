package com.intel.analytics.bigdl.rl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath._
import com.intel.analytics.bigdl.utils.T
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.Engine


/**
  * PGCriterion - Use together with softmax
  * forward returns 0
  * for backward, target contains 2 values, action and reward.
  * gradient = (-1)*(action-prob)*reward
  * use together with sigmoid
  *
  * @param sizeAverage size average of batch
  * @param ev numeric operator
  * @tparam T numeric type
  */

class PGCriterion[@specialized(Float, Double) T: ClassTag]
(
  val isClip:Boolean=false,
  clipP: Double = 1.2,
  clipN: Double = 0.8,
  val sizeAverage: Boolean = true)
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  val clipPos = ev.fromType[Double](clipP)
  val clipNeg = ev.fromType[Double](clipN)

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    //forward is trival here
    output = ev.zero
    output
  }

  def clip (y: T, reward: T): T = {
    var clipped = y
    if (ev.isGreater(reward, ev.zero)) {
      clipped = ev.times(clipPos, y)
      clipped = if (ev.isGreater(clipped, ev.one)) ev.one else clipped
    } else {
      clipped = ev.times(clipNeg, y)
    }
    clipped
  }

  def entropy(input: T, reward: T): T = {
    val target: T = if (isClip) clip(input,reward) else ev.one
    println(s"target=$target")
    ev.negative(ev.times(reward,
      ev.plus(ev.divide(target,input),
        ev.divide(ev.minus(ev.one,target),ev.minus(ev.one,input)))
      )
    )

  }


  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {

    //require(input.dim() == 1 || input.dim() == 2,
    //  "PGCriterion: " +
    //    ErrorInfo.constrainInputAsVectorOrBatch +
    //    s"input dim ${input.dim()}")

    gradInput.resizeAs(input)
    gradInput.zero()

    // if has no batch dim
    if (input.dim() == 1){
      target.squeeze()
      val n = input.size(1)
      require(target.dim()==1,s"target dimension should be 1, but is: ${target.dim()}")
      val targetActIdx = ev.toType[Int](target.valueAt(1))
      val targetReward = target.valueAt(2)
      val inputProb = input.valueAt(targetActIdx)
      val g = entropy(inputProb, targetReward)
      println(s"inputProb=$inputProb, g = $g")
      gradInput.fill(ev.negative(ev.divide(g,ev.fromType[Int](n-1))))
      gradInput.setValue(targetActIdx, g)
    }
    // if contains batch dim
    else if (input.dim() == 2) {
      require(target.dim()==2,s"target dimension should be 2 (with batch dimension, but target dimension is: ${target.dim()}")
      require(target.nElement() == 2*input.size(1),s"target should contain 2*inputsize of elements, but is:${target.nElement()}")

      val batchSize = input.size(1)
      val n = input.size(2)
      // loop
      for (b <- 1 until batchSize + 1) {

        val targetActIdx = ev.toType[Int](target.valueAt(b, 1))
        val targetReward = target.valueAt(b, 2)
        //println(s"$targetActIdx")
        val inputProb = input.valueAt(b, targetActIdx)
        val g = entropy(inputProb,targetReward)
        println(s"g = $g")
        gradInput.select(1, b).fill(ev.negative(ev.divide(g,ev.fromType[Int](n-1))))
        gradInput.setValue(b, targetActIdx, g)
      }

      if (sizeAverage) {
        gradInput.div(ev.fromType[Int](gradInput.nElement()))
      }
    }

    gradInput

  }

}

object PGCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](isClip:Boolean=false,
                                                     clipP: Double = 1.2,
                                                     clipN: Double = 0.8,
                                                     sizeAverage: Boolean = true)
                                                    (implicit ev: TensorNumeric[T]) : PGCriterion[T] = {
    new PGCriterion[T](isClip, clipP, clipN, sizeAverage)
  }
}

package com.intel.analytics.bigdl.rl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class RFPGCriterion[@specialized(Float, Double) T: ClassTag]
(
  val beta : Double = 0.01,
  val epsilon: Double = 1e-8)
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {


  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    //forward is trival here
    output = ev.zero
    // target is of size torch.DoubleTensor(batch_size, envDetails.nbActions)
    // local obj = -2*torch.mean(torch.sum(targets, 2))/numSteps
    // loss needs backward results of target, will extract the logic later.
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {

    // input size should be nxm, n is the batch size
    require(input.dim() == 2, s"input dimension should be 2, the first dimension should be batch size, " +
      s"but actually is: ${target.dim()}")
    // target size should be (n,2)
    // each row is (action, advantage)
    require(target.dim() == 2, s"target dimension should be 2, the first dimension should be batch size, " +
      s"but actually is: ${target.dim()}")

    gradInput.resizeAs(input)
    gradInput.zero()

    //-- add small constant to avoid nans
    val newInput = Tensor[T]().resizeAs(input)
    newInput.copy(input)
    newInput.add(ev.fromType[Double](epsilon))



    val batchSize = newInput.size(1)
    // loop

    for (b <- 1 until batchSize + 1) {
      // ----------------------------------------
      // -- derivative of log categorical w.r.t. p
      // -- d ln(f(x,p))     1/p[i]    if i = x
      // -- ------------ =
      // --     d p          0         otherwise
      // ----------------------------------------
      // lua: targets[i][allActions[i][1]+1] = advantagesNormalized[i] * 1/(output[i][allActions[i][1]+1])
      val targetAction = ev.toType[Int](target.valueAt(b, 1))
      val targetReward = target.valueAt(b, 2)
      println(s"target=$targetAction,$targetReward")
      val targetActProb = newInput.valueAt(b, targetAction)
      val gTargetAct = ev.times(targetReward, ev.divide(ev.one, targetActProb))
      println(s"g = $gTargetAct")
      gradInput.setValue(b, targetAction, gTargetAct)
      println(s"gradInput = $gradInput")
    }
    // -- Add gradEntropy to targets to improve exploration and prevent convergence
    // -- to potentially suboptimal deterministic policy, gradient of entropy of
    //  --policy (for gradient descent): -(-logp(s) - 1)
    val gradEntropy = Tensor[T]().resizeAs(newInput).copy(newInput).log().add(ev.one)
    gradInput.add(ev.fromType[Double](beta), gradEntropy)
    //println(s"gradInput = $gradInput")

    gradInput

  }

}

object RFPGCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](beta : Double = 0.01,
                                                     epsilon: Double = 1e-8)
                                                    (implicit ev: TensorNumeric[T]) : RFPGCriterion[T] = {
    new RFPGCriterion[T](beta, epsilon)
  }
}


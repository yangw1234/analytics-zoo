package com.intel.analytics.bigdl.rl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

class RFPGCriterionSpec extends FlatSpec with Matchers {

  "RFPGCriterion" should "work correctly in batch mode" in {
    val input = Tensor(3,3).fill(0.3333f)
    val target = Tensor(3,2)
    target(Array(1, 1)) = 1f //action
    target(Array(1, 2)) = 0.2f //reward
    target(Array(2, 1)) = 1f //action
    target(Array(2, 2)) = -0.5f //reward
    target(Array(3, 1)) = 2f //action
    target(Array(3, 2)) = 0.6f //reward
    val cr = RFPGCriterion()
    val out = cr.forward(input,target)
    println(s"out = $out")
    val gradIn = cr.backward(input, target)
    println(s"gradIn = $gradIn")
  }

  "RFPGCriterion" should "work correctly in batch mode case 2" in {
    val input = Tensor(2,2)
    input(Array(1,1)) = 0.5f
    input(Array(1,2)) = 0.5f
    input(Array(2,1)) = 0.2f
    input(Array(2,2))=  0.8f
    val target = Tensor(3,2)
    target(Array(1, 1)) = 1f //action
    target(Array(1, 2)) = 5f //reward
    target(Array(2, 1)) = 2f //action
    target(Array(2, 2)) = -5f //reward
    val cr = RFPGCriterion()
    val out = cr.forward(input,target)
    println(s"out = $out")
    val gradIn = cr.backward(input, target)
    println(s"gradIn = $gradIn")
  }

}

package com.intel.analytics.bigdl.rl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}


class PPOCriterionSpec extends FlatSpec with Matchers {

  "PPOCriterion " should "give correct result with dense target" in {
    val criterion = PPOCriterion[Float]()

    val input = Tensor[Float](T(T(0.5, 0.2, 0.3)))
    val target = T(Tensor[Float](T(0.0)), Tensor[Float](T(10.0)) ,Tensor[Float](T(T(0.4, 0.4, 0.2))))


    val output = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)

    println(output)
    println(gradInput)
  }
}

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

  "CPPOCriterion " should "give correct result with dense target" in {
    val criterion = ContinuousPPOCriterion[Double](0.1, 0.0)

    val old_mean = Tensor[Double](T(T(1.0, 2.0, 3.0), T(4.0, 5.0, 6.0)))
    val old_log_std = Tensor[Double](T(T(7.0, 8.0, 9.0), T(10.0, 11.0, 12.0)))
    val curr_mean = Tensor[Double](T(T(13.0, 14.0, 15.0), T(16.0, 17.0, 18.0)))
    val curr_log_std = Tensor[Double](T(T(19.0, 20.0, 21.0), T(22.0, 23.0, 24.0)))

    val ac = Tensor[Double](T(T(25.0, 26.0, 27.0), T(28.0, 29.0, 30.0)))
    val atarg = Tensor[Double](T(31.0, 32.0))

    val input = T(curr_mean, curr_log_std)
    val target = T(ac, atarg, old_mean, old_log_std)


    val output = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)

    println(output)
    println(gradInput)
  }
}

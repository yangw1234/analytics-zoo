/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.examples.tfnet

import scopt.OptionParser

object Options {

  case class TestParams(
                         dataPath: String = "/home/yang/sources/datasets/imagenet/val",
                         modelPath: String = "/tmp/models/tfnet",
                         batchSize: Option[Int] = None,
                         partitionNum: Int = 4
                       )

  val testParser = new OptionParser[TestParams]("BigDL Inception Test Example") {
    opt[String]('f', "folder")
      .text("url of hdfs folder store the hadoop sequence files")
      .action((x, c) => c.copy(dataPath = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = Some(x)))
    opt[Int]('p', "partitionNum")
      .text("num of partitions")
      .action((x, c) => c.copy(partitionNum = x))
  }
}
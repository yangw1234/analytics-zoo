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
package com.intel.analytics.zoo.pipeline.api.net

import java.io.{File, FileInputStream, FileOutputStream, InputStream}
import java.nio._
import java.nio.channels.{Channels, ReadableByteChannel}
import java.nio.file.{Files, Path}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, T}
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.pipeline.api.net.TFNet.TFGraphHolder
import org.tensorflow.framework.GraphDef
import org.tensorflow.types.UInt8
import org.tensorflow.{DataType, Graph, Session, Tensor => TTensor}

import scala.collection.JavaConverters._
import org.json4s._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.control.NonFatal


/**
 * [[TFNet]] wraps a tensorflow subgraph as a layer, and use tensorflow to
 * calculate the layer's output.
 *
 * This subgraph should not contain any tensorflow Variable and the input/output
 * must be numeric types
 *
 * When used with other layers for training, there should be no trainable layer
 * before this one, as the gradInput of this layer is always zero.
 *
 * @param graphDef serialized representation of a graph
 */
class TFNet(private val graphDef: TFGraphHolder,
                    val graphMeta: Meta,
                    config: Array[Int])
  extends AbstractModule[Activity, Activity, Float] with Predictable[Float] {

  protected val module: Module[Float] = this
  implicit val ev = TensorNumeric.NumericFloat
  implicit val tag: ClassTag[Float] = ClassTag.Float

  // todo if an exception is thrown during forward or backward, there will be memory leak
  // maybe create a resource manager to handle tensor creation and destruction

  class ResourceManager() extends java.io.Serializable {
    private var tensorList: List[TTensor[_]] = List()
    def createTFTensor(shape: Array[Long], buffer: FloatBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList = TFTensor :: tensorList
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: ByteBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(classOf[UInt8], shape, buffer)
      tensorList = TFTensor :: tensorList
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: IntBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList = TFTensor :: tensorList
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: LongBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList = TFTensor :: tensorList
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: DoubleBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList = TFTensor :: tensorList
      return TFTensor
    }

    def destructTFTensors(): Unit = {
      for (tensor <- tensorList) {
        tensor.close()
      }
    }
  }

  @transient
  private lazy val tensorManager = new ResourceManager()

  private[zoo] def graph = graphDef.tfGraph.graph

  val inputNames: Array[String] = graphMeta.inputNames
  private val inputTypes = inputNames.map(name2type)

  val outputNames: Array[String] = graphMeta.outputNames
  private val outputTypes = outputNames.map(name2type)
  if (graphMeta.variables.isDefined) {
    // Sanity check. If variables is defined, it means the backward graph
    // is generated. We cannot compute the gradInput/gradWeight if output is not a float
    require(outputTypes.map(_ == DataType.FLOAT).reduce(_ && _),
      "all input types are required to be float if backward are allowed")
  }

  // add Cast Operation if the output tensor is not of type Float
  private val floatOutputNames = outputNames.map { name =>
    val Array(op, idx) = name.split(":")
    val operation = graph.operation(op)
    val output = operation.output(idx.toInt)
    if (output.dataType() != DataType.FLOAT) {
      val name = graph.opBuilder("Cast", s"${op}_to_float")
        .addInput(output)
        .setAttr("DstT", DataType.FLOAT)
        .setAttr("SrcT", output.dataType())
        .build()
        .name()
      s"$name:0"
    } else {
      name
    }
  }

  private val weights = {

    if (graphMeta.variables.isDefined) {
      val ws = new Array[Tensor[Float]](graphMeta.variables.get.length)
        var i = 0
        while (i < ws.length) {
          ws(i) = Tensor[Float]()
          i += 1
        }
      setWeights(ws)
    } else {
      Array[Tensor[Float]]()
    }
  }


  private val gradWeights = {
    if (graphMeta.variables.isDefined) {
      graphMeta.variables.get.map(_ => Tensor[Float]())
    } else {
      Array[Tensor[Float]]()
    }
  }

  private val gradWeightsBuffer = {
    if (graphMeta.variables.isDefined) {
      graphMeta.variables.get.map(_ => Tensor[Float]())
    } else {
      Array[Tensor[Float]]()
    }
  }


  output = {
    if (outputNames.length == 1) {
      Tensor[Float]()
    } else {
      val t = T()
      var i = 0
      while (i < outputNames.length) {
        t.insert(Tensor[Float]())
        i = i + 1
      }
      t
    }
  }

  gradInput = {
    if (inputNames.length == 1) {
      Tensor[Float]()
    } else {
      val t = T()
      var i = 0
      while (i < inputNames.length) {
        t.insert(Tensor[Float]())
        i = i + 1
      }
      t
    }
  }

  @transient
  private[zoo] lazy val sess = {
    val sess = new Session(this.graph, config.map(_.toByte))
    sess
  }
  @transient
  private lazy val inputTFTensors = new Array[TTensor[_]](inputNames.length)
  @transient
  private lazy val weightTFTensors = new Array[TTensor[_]](weights.length)
  @transient
  private lazy val tempTFTensors =
    new Array[TTensor[_]](graphMeta.tempTensors.map(_.length).getOrElse(0))
  @transient
  private lazy val gradWeightTFTensors = new Array[TTensor[_]](gradWeights.length)

  override def updateOutput(input: Activity): Activity = {
    val forwardStartTime = System.nanoTime()
    try {
      val runner = sess.runner()

      require(activityLength(input) == inputTypes.length,
        s"require ${inputTypes.length} inputs, but ${activityLength(input)} given. " +
          s"The inputs are ${inputNames.toSeq}")

      activity2TFTensors(input, inputTypes, inputTFTensors)

      // feed inputs
      inputNames.zipWithIndex.foreach { case (name, idx) =>
        runner.feed(name, inputTFTensors(idx))
      }

      // feed new weights if possible
      graphMeta.variables.map { variableNames =>
        if (! this.isTraining()) {
          var i = 0
          while (i < variableNames.length) {
            if (weightTFTensors(i) == null) {
              val tensor = bigdl2Tf(weights(i), DataType.FLOAT)
              weightTFTensors(i) = tensor
            }
            i += 1
          }
        } else {
          var i = 0
          while (i < variableNames.length) {
            if (weightTFTensors(i) != null) {
              weightTFTensors(i).close()
            }
            val tensor = bigdl2Tf(weights(i), DataType.FLOAT)
            weightTFTensors(i) = tensor
            i += 1
          }
        }
        variableNames.zip(weightTFTensors).map { case (name, tensor) =>
          runner.feed(name, tensor)
          tensor
        }
      }

      // fetch outputs
      floatOutputNames.foreach(runner.fetch)

      // fetch temp tensors used by backward if possible
      if (this.isTraining()) {
        graphMeta.tempTensors.map(_.map(runner.fetch))
      }

      val sessStartTime = System.nanoTime()
      val outputs = runner.run()
      val sessEndTime = System.nanoTime()
      TFNet.logger.debug(s"tensorflow session run ${(sessEndTime - sessStartTime) / 1.0e6}ms")


      outputs.asScala.zipWithIndex.foreach { case (t, idx) =>
        if (idx < outputNames.length) {
          // model outputs
          tf2bigdl(t.asInstanceOf[TTensor[Float]], getOutput(idx + 1))
        } else {
          // temp tensors used by backward if any
          tempTFTensors(idx - outputNames.length) = t
        }
      }
      if (!this.isTraining()) {
        // clean up input tensorflow tensors
        emptyTFTensorArray(tempTFTensors)
      } else {
        // clean up variable tensorflow tensors
        emptyTFTensorArray(weightTFTensors)
      }

      // clean up model output tensorflow tensors
      emptyTFTensorArray(outputs.asScala.slice(0, outputNames.length))
      // tempTensors will be cleaned up after backward
      val forwardEndTime = System.nanoTime()

      TFNet.logger.debug(s"tfnet forward time ${(forwardEndTime - forwardStartTime) / 1.0e6}ms")

      output
    } catch {
      case ex: Throwable =>
        tensorManager.destructTFTensors()
        throw ex
    }
  }

  private def emptyTFTensorArray(arr: Array[TTensor[_]]): Unit = {
    var i = 0
    while (i < arr.length) {
      arr(i).close()
      arr(i) = null
      i += 1
    }
  }

  private def emptyTFTensorArray(arr: mutable.Buffer[TTensor[_]]): Unit = {
    var i = 0
    while (i < arr.length) {
      arr(i).close()
      arr(i) = null
      i += 1
    }
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    try {
      if (graphMeta.variables.isEmpty) {
        generateZeroGrad(input)
      } else {

        val runner = sess.runner()

        require(activityLength(input) == inputTypes.length,
          s"require ${inputTypes.length} inputs, but ${activityLength(input)} given. " +
            s"The inputs are ${inputNames.toSeq}")

        val gradOutputTFTensors = new Array[TTensor[_]](outputNames.length)

        activity2TFTensors(gradOutput, outputTypes, gradOutputTFTensors)

        // feed inputs
        inputNames.zipWithIndex.foreach { case (name, idx) =>
          runner.feed(name, inputTFTensors(idx))
        }

        // feed gradOutputs
        outputNames.map(addGrad).zipWithIndex.foreach { case (name, idx) =>
          runner.feed(name, gradOutputTFTensors(idx))
        }

        // feed temp tensors fetched during forward
        val tempTensorNames = graphMeta.tempTensors.get
        tempTensorNames.zipWithIndex.foreach{ case (name, idx) =>
          runner.feed(name, tempTFTensors(idx))
        }

        // fetch grad inputs
        val gradInputNames = graphMeta.gradInputs.get
        gradInputNames.foreach(runner.fetch)

        // fetch grad weights
        val gradVariableNames = graphMeta.gradVariables.get
        gradVariableNames.foreach(runner.fetch)

        val fetches = runner.run().asScala
        val (i, v) = fetches.splitAt(gradInputNames.length)

        v.map(_.asInstanceOf[TTensor[Float]])
          .zipWithIndex.foreach(x => gradWeightTFTensors(x._2) = x._1)

        i.zipWithIndex.foreach { case (t, idx) =>
          tf2bigdl(t.asInstanceOf[TTensor[Float]], getGradInput(idx + 1))
        }

        // clean up two feeds
        emptyTFTensorArray(inputTFTensors)
        emptyTFTensorArray(gradOutputTFTensors)

        // clean up temp tensors
        emptyTFTensorArray(tempTFTensors)

        // clean up fetched grad inputs
        emptyTFTensorArray(i)

        // grad weights will be cleaned up after acc
      }
      gradInput
    } catch {
      case ex: Throwable =>
        tensorManager.destructTFTensors()
        throw ex
    }
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    try {
      this.gradWeights.zipWithIndex.map { case (gradWeight, idx) =>
        val gradWeightBuffer = this.gradWeightsBuffer(idx)
        val tfTensor = gradWeightTFTensors(idx)
        tf2bigdl(tfTensor, gradWeightBuffer)
        if (gradWeight.isEmpty) {
          gradWeight.resizeAs(weights(idx))
        }
        gradWeight.add(gradWeightBuffer)
      }

      // clean up grad weights tf tensors
      emptyTFTensorArray(gradWeightTFTensors)
    } catch {
      case ex: Throwable =>
        tensorManager.destructTFTensors()
        throw ex
    }
  }

  private def setWeights(weights: Array[Tensor[Float]]) = {
    val runner = sess.runner()
    val variables = graphMeta.variables.get
    variables.foreach(runner.fetch)
    runner.run().asScala.zipWithIndex.map { case (fetch, idx) =>
      val t = weights(idx)
      tf2bigdl(fetch.asInstanceOf[TTensor[Float]], t)
      t
    }
    weights
  }

  override def reset(): Unit = {
    if (graphMeta.variables.isDefined) {
      setWeights(weights)
    }
    zeroGradParameters()
  }

  override def clearState(): this.type = {
    super.clearState()
    gradWeightsBuffer.foreach(_.set())
    this
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, gradWeights)
  }

  override def finalize(): Unit = {
    super.finalize()
    this.sess.close()
  }

  override def release(): Unit = {
    super.release()
    this.sess.close()
  }

  private def getOutput(idx: Int): Tensor[Float] = {
    if (output.isTable) {
      output.toTable[Tensor[Float]](idx)
    } else {
      output.toTensor[Float]
    }
  }

  private def getGradInput(idx: Int): Tensor[Float] = {
    if (gradInput.isTable) {
      gradInput.toTable[Tensor[Float]](idx)
    } else {
      gradInput.toTensor[Float]
    }
  }

  private def name2type(name: String): DataType = {
    val Array(op, idx) = name.split(":")
    val operation = graph.operation(op)
    val output = operation.output(idx.toInt)
    output.dataType()
  }

  private def getShape(names: Seq[String]) = {
    val shapes = names.map { name =>
      val Array(op, idx) = name.split(":")
      val shape = graph.operation(op).output(idx.toInt).shape()
      Shape((0 until shape.numDimensions()).map(shape.size(_).toInt).toArray)
    }

    if (shapes.length == 1) {
      shapes.head
    } else {
      MultiShape(shapes.toList)
    }
  }

  private def bigdl2Tf(t: Tensor[Float], dataType: DataType): TTensor[_] = {

    require(t.isContiguous(), "input to tfnet must be contiguous")
    val shape = t.size().map(_.toLong)
    val arr = t.storage().array()
    val offset: Int = t.storageOffset() - 1
    val length: Int = shape.product.toInt

    if (dataType == DataType.FLOAT) {
      val buffer = FloatBuffer.wrap(arr, offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.UINT8) {
      val buffer = ByteBuffer.wrap(TFNet.floatToUint8(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT32) {
      val buffer = IntBuffer.wrap(TFNet.floatToInt(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT64) {
      val buffer = LongBuffer.wrap(TFNet.floatToLong(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.DOUBLE) {
      val buffer = DoubleBuffer.wrap(TFNet.floatToDouble(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else {
      throw new Exception(s"data type ${dataType} are not supported")
    }


  }

  private def tf2bigdl(t: TTensor[_], output: Tensor[Float]) = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val buffer = FloatBuffer.wrap(
      output.storage().array(),
      output.storageOffset() - 1,
      shape.product)
    t.writeTo(buffer)
  }

  private def activityLength(a: Activity): Int = {
    if (a.isTensor) 1 else a.toTable.length()
  }


  private def activity2TFTensors(input: Activity, types: Seq[DataType],
                                 tfTensors: Array[TTensor[_]]) = {
    if (input.isTensor) {
      require(tfTensors.length == 1, "activity and tfTensors size does not equal," +
        s" activity length is 1 tfTensors length is ${tfTensors.length}")
      val tfTensor = bigdl2Tf(input.toTensor[Float], types.head)
      if (tfTensors(0) != null) {
        tfTensors(0).close()
      }
      tfTensors(0) = tfTensor
    } else {
      val t = input.toTable
      require(tfTensors.length == t.length(), "activity and tfTensors size does not equal," +
        s" activity length is ${t.length()} tfTensors length is ${tfTensors.length}")
      var i = 1
      while (i <= t.length()) {
        val tfTensor = bigdl2Tf(t[Tensor[Float]](i), types(i-1))
        if (tfTensors(i -1) != null) {
          tfTensors(i - 1).close()
        }
        tfTensors(i - 1) = tfTensor
        i += 1
      }
    }
  }

  private def generateZeroGrad(input: Activity) = {
    if (gradInput.isTable) {
      var i = 0
      while (i < gradInput.toTable.length()) {
        gradInput.toTable[Tensor[Float]](i + 1)
          .resizeAs(input.toTable[Tensor[Float]](i + 1))
        i = i + 1
      }
    } else {
      gradInput.toTensor[Float]
        .resizeAs(input.toTensor[Float])
    }
  }

  private def addGrad(name: String) = {
    val parts = name.split(":")
    parts(0) + "_grad:" + parts(1)
  }
}

object TFNet {

  val logger = LoggerFactory.getLogger(getClass)

  private def init(): Unit = {
    val tempDir = Files.createTempDirectory("tensorflow.native.")
    val library_names = Array("iomp5", "mklml_intel",
      "tensorflow_jni", "tensorflow_framework")
    copyAll(tempDir, library_names)
    loadLibrary(library_names, tempDir)
  }

  private def libraryName(name: String) = {
    val os = System.getProperty("os.name").toLowerCase
    var suffix = ".so"
    if (os.contains("mac")) suffix = ".dylib"
    else if (os.contains("win")) suffix = ".dll"
    "lib" + name + suffix
  }

  private def copyAll(tempDir: Path, names: Array[String]) = {
    names.map(libraryName)
      .foreach { lib =>
        val src = resource(lib)
        copyLibraryToTemp(src, lib, tempDir)
        src.close()
      }
  }

  private def resource(name: String) = {
    val resourcePath = "/com/intel/analytics/zoo/pipeline/api/net/" + name
    val url = classOf[TFNet].getResource(resourcePath)
    if (url == null) throw new Error("Can't find the library " + name + " in the resource folder.")
    val in = classOf[TFNet].getResourceAsStream("/com/intel/analytics/zoo/pipeline/api/net/" + name)

    val src = Channels.newChannel(in)
    src
  }

  private def copyLibraryToTemp(src: ReadableByteChannel, name: String, tempDir: Path) = {
    val tempFile = new File(tempDir.toFile + File.separator + name)
    val dst = new FileOutputStream(tempFile)
    try {
      val channel = dst.getChannel
      channel.transferFrom(src, 0, Long.MaxValue)
    } finally {
      dst.close()
    }
  }

  private def loadLibrary(names: Array[String], tempDir: Path) = {
    val prefix = tempDir.toString + File.separator
    names.
      map(prefix + libraryName(_))
      .foreach(System.load)
  }

  if (System.getProperty("analytics-zoo.tensorflow.useMKL") == "true") {
    try {
      init()
      logger.info("MKL TensorFlow loaded")
    } catch {
      case NonFatal(e) =>
        logger.warn("loading mkl TensorFlow failed, fallback to normal TensorFlow", e)
    }
  }


  @transient
  private lazy val inDriver = NetUtils.isDriver

  private val graphRegistry = new RegistryMap[ClosableGraph]()

  private val graphDefRegistry = new RegistryMap[Array[Byte]]()

  class ClosableGraph(val graph: Graph) {
    override def finalize(): Unit = {
      graph.close()
    }
  }

  class TFGraphHolder(@transient var tfGraph: ClosableGraph, private var id: String)
    extends SerializationHolder {

    override def writeInternal(out: CommonOutputStream): Unit = {
      val (graphDef, _) = graphDefRegistry.getOrCreate(id) {
        timing("export as graph def") {
          tfGraph.graph.toGraphDef
        }
      }
      val len = graphDef.length
      out.writeString(id)
      if (inDriver) {
        out.writeInt(len)
        timing(s"writing ${len / 1024 / 1024}Mb graph def to stream") {
          out.write(graphDef)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (graphDef, graphDefIsCreated) = graphDefRegistry.getOrCreate(id) {
        val len = in.readInt()
        require(len != 0, "GraphDef length should not be zero," +
          "please set logging level to debug for more information")
        val graphDef = new Array[Byte](len)
        timing("reading graph def from stream") {
          var numOfBytes = 0
          while (numOfBytes < len) {
            val read = in.read(graphDef, numOfBytes, len - numOfBytes)
            numOfBytes += read
          }
        }
        graphDef
      }

      if (!graphDefIsCreated) {
        val len = in.readInt()
        in.skip(len)
      }

      val (graph, _) = graphRegistry.getOrCreate(id) {
        timing("creating graph obj from graph def") {
          val g = new Graph()
          g.importGraphDef(graphDef)
          new ClosableGraph(g)
        }

      }
      tfGraph = graph
      id = id
    }
  }

  implicit val formats = DefaultFormats

  val defaultSessionConfig = SessionConfig()

  case class SessionConfig(intraOpParallelismThreads: Int = 1,
                           interOpParallelismThreads: Int = 1,
                           usePerSessionThreads: Boolean = true,
                           graphRewriteRemapping: Boolean = false) {

    // Ideally we should use the following code, however, importing tensorflow proto
    // will conflict with bigdl.

    def toByteArray(): Array[Byte] = {
      val intraSeq = if (intraOpParallelismThreads > 0) {
        Seq(16, intraOpParallelismThreads)
      } else {
        Seq[Int]()
      }
      val interSeq = if (interOpParallelismThreads > 0) {
        Seq(40, interOpParallelismThreads)
      } else {
        Seq[Int]()
      }
      val perSessSeq = if (usePerSessionThreads) {
        Seq(72, 1)
      } else {
        Seq[Int]()
      }

      val reWriteSess = if (graphRewriteRemapping) {
        Seq[Int]()
      } else {
        Seq[Int](82, 4, 82, 2, 112, 2)
      }

      (intraSeq ++ interSeq ++ perSessSeq ++ reWriteSess).map(_.toByte).toArray
    }
  }


  private def floatToInt(array: Array[Float]): Array[Int] = {
    val result = new Array[Int](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toInt
      i = i + 1
    }
    result
  }

  private def floatToLong(array: Array[Float]): Array[Long] = {
    val result = new Array[Long](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toLong
      i = i + 1
    }
    result
  }

  private def floatToDouble(array: Array[Float]): Array[Double] = {
    val result = new Array[Double](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toDouble
      i = i + 1
    }
    result
  }

  private def floatToUint8(array: Array[Float]): Array[Byte] = {
    val result = new Array[Byte](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toByte
      i = i + 1
    }
    result
  }

  /**
   * Create a TFNet
   * @param graphDef the tensorflow GraphDef object
   * @return
   */
  private[zoo] def apply(graphDef: GraphDef, graphId: String,
                    graphMeta: Meta,
                    config: Array[Byte]): TFNet = {
    val graph = new Graph()
    graph.importGraphDef(graphDef.toByteArray)

    new TFNet(new TFGraphHolder(new ClosableGraph(graph), graphId), graphMeta, config.map(_.toInt))
  }

  /**
   * Create a TFNet
   * @param path the file path of a graphDef
   * @param inputNames the input tensor names of this subgraph
   * @param outputNames the output tensor names of this subgraph
   * @return
   */
  def apply(path: String,
            inputNames: Array[String],
            outputNames: Array[String],
            config: SessionConfig): TFNet = {
    val graphDef = parseGraph(path)
    val graphMeta = Meta(inputNames = inputNames, outputNames = outputNames)
    TFNet(graphDef, path, graphMeta, config.toByteArray())
  }

  /**
   * Create a TFNet
   * @param path the file path of a graphDef
   * @param inputNames the input tensor names of this subgraph
   * @param outputNames the output tensor names of this subgraph
   * @return
   */
  def apply(path: String,
            inputNames: Array[String],
            outputNames: Array[String]): TFNet = {
    TFNet(path, inputNames, outputNames, defaultSessionConfig)
  }


  def apply(folder: String, config: SessionConfig = TFNet.SessionConfig()): TFNet = {
    val (model, meta) = NetUtils.processTFFolder(folder)
    val graphDef = parseGraph(model)
    TFNet(graphDef, model, meta, config.toByteArray())
  }

  private[zoo] def parseGraph(graphProtoTxt: String) : GraphDef = {
    var fr: File = null
    var in: InputStream = null
    try {
      fr = new File(graphProtoTxt)
      in = new FileInputStream(fr)

      GraphDef.parseFrom(in)
    } finally {
      if (in != null) in.close()
    }
  }
}

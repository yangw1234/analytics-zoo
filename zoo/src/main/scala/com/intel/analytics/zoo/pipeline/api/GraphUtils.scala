package com.intel.analytics.bigdl

import com.intel.analytics.bigdl.nn.{DynamicGraph, Graph, StaticGraph}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node

import scala.collection.mutable
import scala.reflect.ClassTag


object GraphUtils {

  type GraphWithUtils[T] = Graph[T] with GraphUtils[T, _ <: Graph[T] with GraphUtils[T, _]]

  def getOutputs[T](graph: Graph[T]): Seq[ModuleNode[T]] = {
    graph.outputs
  }

  def withGraphUtils[T: ClassTag](graph: Graph[T])
        (implicit ev: TensorNumeric[T]): Graph[T] with GraphUtils[T, _ <: Graph[T] with GraphUtils[T, _]] = {
    val inputs = graph.inputs
    val outputs = graph.outputs
    val variables = graph.variables

    graph match {

      case g: StaticGraph[T] =>
        new StaticGraphWithUtils[T](inputs, outputs, variables)
      case g: DynamicGraph[T] =>
        new DynamicGraphWithUtils[T](inputs, outputs, variables, g.generateBackward)
    }
  }

}

class StaticGraphWithUtils[T: ClassTag] (
   private val _inputs : Seq[ModuleNode[T]],
   private val _outputs : Seq[ModuleNode[T]],
   private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)(implicit ev: TensorNumeric[T])
  extends StaticGraph[T](_inputs, _outputs, _variables) with GraphUtils[T, StaticGraphWithUtils[T]] {

  override def newGraph(output: String): StaticGraphWithUtils[T] = {
    new StaticGraphWithUtils[T](inputs, nodes(Seq(output)), _variables)
  }

  override def newGraph(outputs: Seq[String]): StaticGraphWithUtils[T] = {
    new StaticGraphWithUtils[T](inputs, nodes(outputs))
  }

  override def getInputs: Seq[ModuleNode[T]] = inputs

  override def getOutputs: Seq[ModuleNode[T]] = outputs
}

class DynamicGraphWithUtils[T: ClassTag](
  _inputs : Seq[ModuleNode[T]],
  _outputs : Seq[ModuleNode[T]],
  _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None,
  generateBackward: Boolean = true)(implicit ev: TensorNumeric[T])
  extends DynamicGraph[T](_inputs, _outputs, _variables, generateBackward) with GraphUtils[T, DynamicGraphWithUtils[T]] {

  override def newGraph(output: String): DynamicGraphWithUtils[T] = {
    new DynamicGraphWithUtils[T](
      inputs,
      nodes(Seq(output)).map(_.removeNextEdges()),
      _variables, generateBackward)
  }

  override def newGraph(outputs: Seq[String]): DynamicGraphWithUtils[T] = {
    new DynamicGraphWithUtils[T](
      inputs,
      nodes(outputs).map(_.removeNextEdges()), _variables, generateBackward)
  }

  override def getInputs: Seq[ModuleNode[T]] = inputs

  override def getOutputs: Seq[ModuleNode[T]] = outputs

}


trait GraphUtils[T, D <: Module[T] with GraphUtils[T, D]] {

  def nodes(names: Seq[String]): Seq[ModuleNode[T]] = {
    names.map(node)
  }

  def node(name: String): ModuleNode[T]

  def getInputs: Seq[ModuleNode[T]]

  def getOutputs: Seq[ModuleNode[T]]

  def freezeUpTo(names: String*): this.type = {
    DFS(nodes(names)).foreach(_.element.freeze())
    this
  }

  def newGraph(output: String): D

  def newGraph(outputs: Seq[String]): D

  private def DFS(endPoints: Seq[ModuleNode[T]]): Iterator[ModuleNode[T]] = {
    new Iterator[ModuleNode[T]] {
      private val stack = new mutable.Stack[ModuleNode[T]]()
      endPoints.map(stack.push)
      private val visited = new mutable.HashSet[ModuleNode[T]]()

      override def hasNext: Boolean = stack.nonEmpty

      override def next(): ModuleNode[T] = {
        require(hasNext, "No more elements in the graph")
        val node = stack.pop()
        visited.add(node)
        val nextNodes = node.prevNodes
        // to preserve order
        val nodesSet = mutable.LinkedHashSet[ModuleNode[T]]()
        nextNodes.foreach(nodesSet.add)
        nodesSet.filter(!visited.contains(_))
          .filter(!stack.contains(_)).foreach(stack.push)
        node
      }
    }
  }

}
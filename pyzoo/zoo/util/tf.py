#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging

from tensorflow.python.ops import gen_state_ops, gen_resource_variable_ops

from bigdl.util.tf_utils import save_variable_bigdl
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util, tensor_util
from tensorflow.python.framework import ops
from tensorflow.python.framework.graph_util_impl import extract_sub_graph
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import json
import copy


def process_grad(grad, tensor, allow_non_differentiable):

    if grad is not None:
        grad = ops.convert_to_tensor_or_indexed_slices(grad)
        if isinstance(grad, ops.IndexedSlices):
            # In IndexedSlices is not supported in java api, we have to convert it to
            # a dense tensor. This operation is potentially expensive, but there seems
            # no work around
            grad = tf.unsorted_segment_sum(grad.values, grad.indices,
                                           grad.dense_shape[0])
    else:
        if allow_non_differentiable:
            grad = tf.zeros(shape=tf.shape(tensor))
        else:
            raise ValueError("tensor: %s is not differentiable" % tensor.name)
    return grad


def export_tf_without_freezing(sess, folder, inputs, outputs, variables, trainable_variables, backward_info=None):
    output_names = [o.name for o in outputs]
    input_names = [i.name for i in inputs]
    variable_names = [v.name for v in variables]
    trainable_variable_names = [v.name for v in trainable_variables]

    returned_variables = []

    for v in variables:
        if isinstance(v, tf.Variable):
            returned_variables.append(sess.run(v))
        elif v.dtype == tf.resource:
            new_name = v.name[:-2] + "/Read/ReadVariableOp:0"
            returned_variables.append(sess.run(new_name))
        else:
            returned_variables.append(sess.run(v.name))

    graph_def = tf.get_default_graph().as_graph_def()

    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name="")

        out_variables = [g.get_tensor_by_name(name) for name in variable_names]
        trainable_set = set(trainable_variable_names)

        assigns = []
        train_assigns = []
        for i in range(len(out_variables)):
            v = out_variables[i]
            p = tf.placeholder(dtype=tf.float32, shape=returned_variables[i].shape, name=v.name.split(":")[0] + "_assign")
            p = tf.cast(p, dtype=returned_variables[i].dtype)
            if v.dtype == tf.resource:
                a = gen_resource_variable_ops.assign_variable_op(
                    v,
                    p)
            else:
                a = tf.assign(v, p)
            if v.name in trainable_set:
                train_assigns.append(a)
            assigns.append(a)
        assign = tf.group(*assigns)
        train_assign = tf.group(*train_assigns)
        with gfile.GFile(os.path.join(folder, "graph.pb"), "wb") as f:
            f.write(g.as_graph_def().SerializeToString())

        meta = {
            "input_names": input_names,
            "output_names": output_names,
        }

        if len(variables) > 0:
            variable_dict = dict(zip(variable_names, returned_variables))
            save_variable_bigdl(variable_dict, folder + "/model.bin")
            meta["variables"] = variable_names
            meta["variable_path"] = folder + "/model.bin"
            meta["trainable_variables"] = trainable_variable_names
            meta["assign_op"] = assign.name
            meta["train_assign_op"] = train_assign.name

        if backward_info is not None:
            meta.update(backward_info)
        else:
            meta["has_backward"] = False

        with open(os.path.join(folder, "graph_meta.json"), "w") as f:
            f.write(json.dumps(meta))


def generate_backward_graph(graph_def, input_names,
                            output_names, variable_names,
                            allow_non_differentiable):
    nodes = set([n.name for n in graph_def.node])
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name='')
        output_tensors = [g.get_tensor_by_name(x) for x in output_names]
        grad_output_placeholders = [tf.placeholder(dtype=x.dtype,
                                                   name=x.name.split(":")[0] + "_grad",
                                                   shape=x.shape) for x in output_tensors]

        variables = [g.get_tensor_by_name(x) for x in variable_names]

        inputs = [g.get_tensor_by_name(x) for x in input_names]
        variables_and_inputs = variables + inputs
        grads = tf.gradients(output_tensors, variables_and_inputs,
                             grad_ys=grad_output_placeholders)

        grads = [process_grad(grad, tensor, allow_non_differentiable)
                 for (grad, tensor) in zip(grads, variables_and_inputs)]

        temp_tensor_names = _find_temp_tensors(grads, nodes, input_names)

        grad_variable_names = [x.name for x in grads[0:len(variables)]]
        grad_input_names = [x.name for x in grads[len(variables):]]

        new_graph_def = g.as_graph_def()
    return new_graph_def, grad_variable_names, grad_input_names, temp_tensor_names


def export_tf(sess, folder, inputs, outputs,
              generate_backward=False, allow_non_differentiable=True):
    """
    Export the frozen tensorflow graph as well as the inputs/outputs information
    to the folder for inference.

    This function will
    1. freeze the graph (replace all variables with constants)
    2. strip all unused node as specified by inputs and outputs
    3. add placeholder nodes as needed
    4. write the frozen graph and inputs/outputs names to the folder

    Note: There should not be any queuing operation between inputs and outputs

    :param sess: tensorflow session holding the variables to be saved
    :param folder: the folder where graph file and inputs/outputs information are saved
    :param inputs: a list of tensorflow tensors that will be fed during inference
    :param outputs: a list of tensorflow tensors that will be fetched during inference
    :return:
    """

    output_node_names = list({t.op.name for t in outputs})

    graph_def = sess.graph_def
    graph = sess.graph

    # clear device specifications
    for node in graph_def.node:
        node.device = ""

    non_placeholder_input_names = []
    type_enums = []
    for input_tensor in inputs:
        if input_tensor.op.type != "Placeholder":
            non_placeholder_input_names.append(input_tensor.name)
            type_enums.append(input_tensor.dtype.as_datatype_enum)

    output_names = [o.name for o in outputs]

    all_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # freeze graph
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        output_node_names
    )

    optimized_graph_def, old_names2new = strip_unused(frozen_graph_def,
                                                      non_placeholder_input_names,
                                                      output_names,
                                                      type_enums)

    nodes_of_graph = []
    for node in optimized_graph_def.node:
        nodes_of_graph.append(node.name + ":0")
    nodes_of_graph_set = set(nodes_of_graph)

    new_input_names = []
    error_input_nodes = []
    for t in inputs:
        if t.name in old_names2new:
            if old_names2new[t.name] not in nodes_of_graph_set:
                error_input_nodes.append("\"" + (t.name)[0:-2] + "\"")
            new_input_names.append(old_names2new[t.name])
        else:
            if t.name not in nodes_of_graph_set:
                error_input_nodes.append("\"" + (t.name)[0:-2] + "\"")
            new_input_names.append(t.name)

    if error_input_nodes:
        error_nodes_name = " and ".join(error_input_nodes)
        raise ValueError("Node %s doesn't exist in the graph" % str(error_nodes_name))

    # check all placeholder in the graph are listed in the new_input_names:
    new_input_nodes = {name.split(":")[0] for name in new_input_names}
    for node in optimized_graph_def.node:
        if node.op == "Placeholder" and node.name not in new_input_nodes:
            raise ValueError(
                "Node %s is a Placeholder but not listed in inputs, inputs are %s"
                % (node.name, inputs))

    temp_tensors = None
    used_variables = []
    grad_variables = []
    grad_inputs = []
    if generate_backward:
        nodes = set([n.name for n in optimized_graph_def.node])
        for v in all_variables:
            if v.op.name in nodes:
                used_variables.append(v.name)

        with tf.Graph().as_default() as g:
            tf.import_graph_def(optimized_graph_def, name='')
            output_tensors = [g.get_tensor_by_name(x) for x in output_names]
            grad_output_placeholders = [tf.placeholder(dtype=x.dtype,
                                                       name=x.name.split(":")[0] + "_grad",
                                                       shape=x.shape) for x in output_tensors]

            variables = [g.get_tensor_by_name(x) for x in used_variables]

            inputs = [g.get_tensor_by_name(x) for x in new_input_names]
            grads = tf.gradients(output_tensors, variables + inputs,
                                 grad_ys=grad_output_placeholders)

            grads = [process_grad(grad, tensor, allow_non_differentiable)
                     for (grad, tensor) in zip(grads, variables + inputs)]

            temp_tensors = _find_temp_tensors(grads, nodes, new_input_names)

            grad_variables = [x.name for x in grads[0:len(variables)]]
            grad_inputs = [x.name for x in grads[len(variables):]]

            optimized_graph_def = g.as_graph_def()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with gfile.GFile(os.path.join(folder, "frozen_inference_graph.pb"), "wb") as f:
        f.write(optimized_graph_def.SerializeToString())

    meta = {
        "input_names": new_input_names,
        "output_names": output_names
    }

    if generate_backward:
        meta["temp_tensors"] = list(temp_tensors)
        meta["variables"] = used_variables
        meta["grad_variables"] = grad_variables
        meta["grad_inputs"] = grad_inputs

    with open(os.path.join(folder, "graph_meta.json"), "w") as f:
        f.write(json.dumps(meta))


def _find_temp_tensors(grads, forward_ops, input_names):
    '''
    find all the tensors that are used for computing grads and has been
    computed during forward
    :param grads:
    :param forward_ops:
    :return:
    '''
    import sys
    is_py2 = sys.version[0] == '2'
    if is_py2:
        import Queue as queue
    else:
        import queue as queue
    queue = queue.Queue()
    for grad in grads:
        queue.put(grad)

    temp_tensors = set()
    visited = set()
    while not queue.empty():
        tensor = queue.get()
        # this is necessary, because input may not be differentiable
        if tensor is None:
            continue
        else:
            visited.add(tensor.name)
            if tensor.name in input_names:
                continue
            if tensor.op.name in forward_ops:
                temp_tensors.add(tensor.name)
                continue
            for input_tensor in tensor.op.inputs:
                # this is necessary because there may be a cycle in the graph such as tf.while_loop
                if input_tensor.name not in visited:
                    queue.put(input_tensor)
    return temp_tensors


def strip_unused(input_graph_def, input_tensor_names, output_tensor_names,
                 placeholder_type_enum):
    """Removes unused nodes from a GraphDef.

  Args:
    input_graph_def: A graph with nodes we want to prune.
    input_tensor_names: A list of the nodes we use as inputs.
    output_tensor_names: A list of the output nodes.
    placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.

  Returns:
    A `GraphDef` with all unnecessary ops removed. and a map containing the old input
    names to the new input names

  Raises:
    ValueError: If any element in `input_node_names` refers to a tensor instead
      of an operation.
    KeyError: If any element in `input_node_names` is not found in the graph.
  """
    for name in input_tensor_names:
        if ":" not in name:
            raise ValueError("Input '%s' appears to refer to a Operation, "
                             "not a Tensor." % name)

    old2new = {}

    # Here we replace the nodes we're going to override as inputs with
    # placeholders so that any unused nodes that are inputs to them are
    # automatically stripped out by extract_sub_graph().
    not_found = {name for name in input_tensor_names}
    input_node_names = {name.split(":")[0] for name in input_tensor_names}
    output_node_names = list({name.split(":")[0] for name in output_tensor_names})
    inputs_replaced_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name not in input_node_names:
            for i in range(len(node.input)):
                if _append_port(node.input[i]) in input_tensor_names:
                    old_name = _append_port(node.input[i])
                    not_found.remove(old_name)
                    new_input_name = node.input[i].replace(":", "_")
                    placeholder_node = node_def_pb2.NodeDef()
                    placeholder_node.op = "Placeholder"
                    placeholder_node.name = new_input_name
                    if isinstance(placeholder_type_enum, list):
                        input_node_index = input_tensor_names.index(old_name)
                        placeholder_node.attr["dtype"].CopyFrom(
                            attr_value_pb2.AttrValue(type=placeholder_type_enum[
                                input_node_index]))
                    else:
                        placeholder_node.attr["dtype"].CopyFrom(
                            attr_value_pb2.AttrValue(type=placeholder_type_enum))
                    if "_output_shapes" in node.attr:
                        placeholder_node.attr["_output_shapes"].CopyFrom(
                            node.attr["_output_shapes"])
                    node.input[i] = new_input_name
                    old2new[old_name] = new_input_name + ":0"
                    inputs_replaced_graph_def.node.extend([placeholder_node])
            inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])

    if not_found:
        raise KeyError("The following input nodes were not found: %s\n" % not_found)

    output_graph_def = graph_util.extract_sub_graph(inputs_replaced_graph_def,
                                                    output_node_names)
    return output_graph_def, old2new


def _append_port(input_name):
    if input_name.find(":") == -1:
        return input_name + ":0"
    else:
        return input_name

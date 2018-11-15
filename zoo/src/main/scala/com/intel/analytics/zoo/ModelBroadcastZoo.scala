package com.intel.analytics.bigdl

import com.intel.analytics.bigdl.models.utils.{CachedModels, ModelBroadcast, ModelBroadcastImp, ModelInfo}
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Util._
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag


// this is only for experiment
class ModelBroadcastZoo[T: ClassTag](applyProtoBuffer: Boolean = false)
                                                   (implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {

  private val singleThead = Engine.coreNumber() == 1

  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _

  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    CachedModels.deleteAll(uuid) // delete the models on driver

    if (applyProtoBuffer) {
      broadcastModel = sc.broadcast(ModelInfo(uuid, model))
    } else {
      // broadcast Consts
      if (model.isInstanceOf[Container[_, _, T]]) {
        val moduleConsts = getAndClearConsts(model.asInstanceOf[Container[_, _, T]])
        // TODO: broadcast Const, model structure and weight in the same broadcast.
        broadcastConsts = sc.broadcast(moduleConsts)
      }
      // broadcast weight and model
      val weightsBias = getAndClearWeightBias(model.parameters())
      broadcastModel = sc.broadcast(ModelInfo[T](uuid, model))
      broadcastParameters = sc.broadcast(weightsBias)

      // For quantized model if we don't clone weightsBias, the original model will be released also
      // when we delete all models used in `ModelBroadcast`.
      putWeightBias(SerializationUtils.clone(weightsBias), model)
      initGradWeightBias(weightsBias, model)
    }
    this
  }

  /**
    * get the broadcast model
    * put the weight and bias back to the model
    *
    * @param initGradient If create a tensor for gradient when fetch the model. Please note that
    *                     the gradient is not needed in model inference
    * @return model
    */
  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
    CachedModels.deleteAll(uuid)
    if (applyProtoBuffer) {
      val localModel = broadcastModel.value.model.clone(false)
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)

      if (initGradient) {
        initGradWeightBias(getWeightBias(localModel.parameters()), localModel)
      }
      localModel
    } else {
      if (this.singleThead) {
        return broadcastModel.value.model
      }
      val localModel = broadcastModel.value.model.cloneModule()
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)

      val parameters = if (shareWeight) {
        broadcastParameters.value
      } else {
        SerializationUtils.clone(broadcastParameters.value)
      }

      // share weight
      putWeightBias(parameters, localModel)
      // share Consts
      if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
        putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
      }
      // init gradient
      if (initGradient) {
        initGradWeightBias(broadcastParameters.value, localModel)
      }
      localModel
    }
  }

  private def getWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]]))
  : Array[Tensor[T]] = {
    if (parameters._1.length != 0) {
      var i = 0
      val weightsBias = new Array[Tensor[T]](parameters._1.length)
      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
      val (isCompacted, storage) = if (!isQuantized) {
        val storage = Storage(parameters._1(0).storage.array())
        (parameters._1.map(_.nElement()).sum == storage.length(), storage)
      } else {
        (false, null)
      }

      // get weight and bias
      while (i < parameters._1.length) {
        if (parameters._1(i) != null) {
          val wb = parameters._1(i)
          wb.getTensorType match {
            case QuantizedType =>
              val quantTensor = wb.asInstanceOf[QuantizedTensor[T]]
              weightsBias(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            case _ =>
              weightsBias(i) = if (isCompacted) {
                Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
              } else {
                Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
              }
          }
          i += 1
        }
      }
      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }
}

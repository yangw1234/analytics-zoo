import tensorflow as tf


class Iterator(object):

    def __init__(self, next_tensors):
        self.next_tensors = next_tensors

    def get_next(self):
        return self.next_tensors

class Dataset(object):

    """
    Represents a distributed set of elements backed by an RDD,
    which is created by applying tensorflow dataset transformations
    on each partitions.
    """

    create_dataset = None
    rdd = None
    driver_tf_dataset = None
    next_tensors = None
    batch_size = None

    def __init__(self):
        pass

    @staticmethod
    def from_tensor_slices(sc, tensors):
        return TensorSliceDataset(sc, tensors)

    @staticmethod
    def from_rdd(rdd, output_types, output_shapes=None):
        dataset = Dataset()
        dataset.rdd = rdd

        def create_dataset_fn(ts):
            def gen():
                for t in ts:
                    yield t
            return tf.data.Dataset.from_generator(gen, output_types, output_shapes=output_shapes)

        def driver_gen():
            for t in [rdd.collect()]:
                yield t
        dataset.create_dataset = create_dataset_fn
        dataset.driver_tf_dataset = tf.data.Dataset.from_generator(driver_gen, output_types, output_shapes)
        return dataset

    @property
    def output_classes(self):
        if self.driver_tf_dataset is None:
            raise ValueError("subclass must set driver_tf_dataset")

        return self.driver_tf_dataset.output_classes

    @property
    def output_shapes(self):
        if self.driver_tf_dataset is None:
            raise ValueError("subclass must set driver_tf_dataset")

        return self.driver_tf_dataset.output_shapes

    @property
    def output_types(self):
        if self.driver_tf_dataset is None:
            raise ValueError("subclass must set driver_tf_dataset")

        return self.driver_tf_dataset

    def make_one_shot_iterator(self):
        return Iterator(self.get_next_tensors())

    def batch(self, batch_size):
        return BatchDataset(self, batch_size)

    def map(self, map_func):

        return MapDataset(self, map_func)

    def filter(self, predicate):

        return FilterDataset(self, predicate)

    def as_rdd(self):

        create_dataset_func = self.create_dataset

        def map_partitions_func(iterator, dataset_func):

            dataset = dataset_func(iterator)
            iter = dataset.make_one_shot_iterator()
            next = iter.get_next()
            result = []
            with tf.Session() as sess:
                while True:
                    try:
                        value = sess.run(next)
                        result.append(value)
                    except tf.errors.OutOfRangeError:
                        return result

        return self.rdd.mapPartitions(lambda iter: map_partitions_func(iter, create_dataset_func))

    def get_next_tensors(self):
        if self.next_tensors is None:
            self.next_tensors = self.driver_tf_dataset.make_one_shot_iterator().get_next()
        return self.next_tensors


class TensorSliceDataset(Dataset):

    def __init__(self, sc, tensors):

        super(TensorSliceDataset, self).__init__()

        self.driver_tf_dataset = tf.data.Dataset.from_tensor_slices(tensors)

        output_types = self.driver_tf_dataset.output_types
        output_shapes = self.driver_tf_dataset.output_shapes

        def create_dataset_fn(ts):
            def gen():
                for t in ts:
                    yield t
            return tf.data.Dataset.from_generator(gen, output_types, output_shapes=output_shapes)

        self.create_dataset = create_dataset_fn

        next = self.driver_tf_dataset.make_one_shot_iterator().get_next()
        result = []
        with tf.Session() as sess:
            while True:
                try:
                    value = sess.run(next)
                    result.append(value)
                except tf.errors.OutOfRangeError:
                    break

        self.rdd = sc.parallelize(result)


class MapDataset(Dataset):

    def __init__(self, input_dataset, map_func):

        super(MapDataset, self).__init__()

        self.rdd = input_dataset.rdd
        self.batch_size = input_dataset.batch_size

        create_pre_datset = input_dataset.create_dataset

        def create_dataset_fn(ts):
            dataset = create_pre_datset(ts)
            return dataset.map(map_func)

        self.create_dataset = create_dataset_fn
        self.driver_tf_dataset = input_dataset.driver_tf_dataset.map(map_func)


class FilterDataset(Dataset):

    def __init__(self, input_dataset, predicate):

        super(FilterDataset, self).__init__()

        self.rdd = input_dataset.rdd
        self.batch_size = input_dataset.batch_size

        create_pre_dataset = input_dataset.create_dataset

        def create_dataset_fn(ts):
            dataset = create_pre_dataset(ts)
            return dataset.filter(predicate)

        self.create_dataset = create_dataset_fn
        self.driver_tf_dataset = input_dataset.driver_tf_dataset.map(predicate)


class BatchDataset(Dataset):

    def __init__(self, input_dataset, batch_size):
        super(BatchDataset, self).__init__()
        self.rdd = input_dataset.rdd
        self.batch_size = batch_size
        create_pre_dataset = input_dataset.create_dataset

        def create_dataset_fn(ts):
            dataset = create_pre_dataset(ts)
            return dataset.batch(1)
        self.create_dataset = create_dataset_fn
        self.driver_tf_dataset = input_dataset.driver_tf_dataset.batch(batch_size)


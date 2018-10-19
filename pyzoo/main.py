import numpy

from model import StandardModel
from nmt import parse_args, init_or_restore_variables, read_all_lines
import tensorflow as tf
import util
from zoo import init_nncontext
from zoo.common import Sample

from zoo.pipeline.api.net import TFNet
import os

# os.environ['PYSPARK_SUBMIT_ARGS'] = "--driver-java-options \" -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=5050\" /home/yang/sources/zoo/pyzoo/main.py --dictionaries /home/yang/sources/nematus/test/data/vocab.en.json /home/yang/sources/nematus/test/data/vocab.de.json --datasets /home/yang/sources/nematus/test/data/corpus.en /home/yang/sources/nematus/test/data/corpus.de --valid_source_dataset /home/yang/sources/nematus/test/data/corpus.en"


def preprocess(x):
    y_dummy = numpy.zeros(shape=(len(x), 1))
    x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                        maxlen=None)

    x_in = numpy.repeat(x, repeats=2, axis=-1)
    x_mask_in = numpy.repeat(x_mask, repeats=2, axis=-1)
    x_in = x_in.transpose((2, 1, 0))
    x_mask_in = x_mask_in.transpose()
    return [x_in, x_mask_in]

def to_sample(features):
    return Sample.from_ndarray(features, numpy.zeros([features[0].shape[0]]))


if __name__ == "__main__":

    config = parse_args()
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    with tf.Session(config=tf_config) as sess:

        model = StandardModel(config)
        beam_ys, parents, cost = model._get_beam_search_outputs(2)
        saver = init_or_restore_variables(config, sess)
        print model.inputs.x.shape
        print model.inputs.x_mask.shape
        print config.beam_size
        inputs = [model.inputs.raw_x, model.inputs.raw_x_mask]
        outputs = [beam_ys, parents, cost]
        tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)

        sentences = open(config.valid_source_dataset, 'r').readlines()
        batches, idxs = read_all_lines(config, sentences, 1)
                                           #config.valid_batch_size)

        sc = init_nncontext()

        data_rdd = sc.parallelize(batches[:4]).map(lambda x: preprocess(x)).map(lambda x: to_sample(x))

        result_rdd = tfnet.predict(data_rdd, batch_pre_core=1)

        print result_rdd.collect()
        beams = []
        for x in batches:
            y_dummy = numpy.zeros(shape=(len(x), 1))
            x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                                maxlen=None)

            x_in = numpy.repeat(x, repeats=2, axis=-1)
            x_mask_in = numpy.repeat(x_mask, repeats=2, axis=-1)
            print x_in.shape
            print x_mask_in.shape
            sample = tfnet.forward([x_in, x_mask_in])
            # sample = sess.run(outputs, {model.inputs.x:x_in, model.inputs.x_mask:x_mask_in})
            beams.extend(sample)
            break

        # print beams

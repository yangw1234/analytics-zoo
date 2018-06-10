import itertools
import re
from optparse import OptionParser

from bigdl.dataset import news20
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample
import datetime as dt
import numpy as np
import pickle
from time import time
from pyspark import SparkFiles, SparkContext
from pyspark.sql import Row

from zoo import init_nncontext
from zoo.pipeline.api.net import TFNet


def load_word2id(path1):
    return pickle.load(open(path1, "r"))

def _list_to_word_ids(lineLists, word_to_id,max_sentence_length):
    unkown_word_id = len(word_to_id)
    padding_word_id = len(word_to_id)+1
    word_ids = []
    count = 0
    for word in lineLists:
        if count==max_sentence_length:
            break
        if word in word_to_id:
            word_ids.append(word_to_id[word])
        else:
            word_ids.append(unkown_word_id)
        count += 1
    while count < max_sentence_length:
        count += 1
        word_ids.append(padding_word_id)
    return word_ids

def split_line(line):
    data = line.split("\t")
    return (data[0].split(" "), int(data[1].split(" ")[0]))
    
def words_to_ids(words, word2id, max_sentence_length):
    return _list_to_word_ids(words, word2id, max_sentence_length)
    
def to_sample(input1, input2, label):  
    input1 = JTensor.from_ndarray(np.array(input1))
    # bigdl's lable is [1 - n]
    return Sample.from_jtensor([input1, input2], label + 1)

def to_predict_sample(line, word_to_id, max_sentence_length, drop_tensor):
    words = line.split(" ")
    ids = words_to_ids(words, word_to_id, max_sentence_length)
    sample = to_sample(ids, drop_tensor, 1) # add a fake label
    return sample
    

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="test")
    parser.add_option("-d", "--data_path", dest="data_path", default="/home/yang/sources/datasets/sougou/data/test1000.txt")
    parser.add_option("-o", "--modelPath", dest="modelPath", default="/tmp/text_out/")
    parser.add_option("-t", "--output", dest="outputPath", default="/tmp/text_out/")
    parser.add_option("-m", "--max_sentence_length", dest="max_sentence_length", default=40)
    parser.add_option("-b", "--batchsize", dest="batchsize", default=128)
    parser.add_option("-w", "--word2id", dest="word2id", default="/home/yang/sources/datasets/sougou/data/word2id")
    parser.add_option("-e", "--word_embeddings", dest="word_embeddings", default="/home/yang/sources/datasets/sougou/data/word_embeddings")
    (options, args) = parser.parse_args(sys.argv)
    
    word2id_path = options.word2id
    max_sentence_length = int(options.max_sentence_length)
    batch_size = int(options.batchsize)
    data_path = options.data_path
    model_path = options.modelPath
    output_path = options.outputPath
    
    print "Option parsing finished."
    
    # # create spark conf
    # conf = create_spark_conf()
    # sc = SparkContext(appName="short_text_classifier", conf=conf)

    sc = init_nncontext()
    
    if options.action == "train":
        pass
    elif options.action == "test":
        # load word2id
        sc.addFile(word2id_path)
        word2id = load_word2id(word2id_path)
        
        # create broadcast variable
        drop_percent = 1.0
        drop = JTensor.from_ndarray(np.full([300], drop_percent))
        broadcast_drop = sc.broadcast(drop)
        broadcast_word2id = sc.broadcast(word2id)
        
        # create test_rdd
        test_rdd = sc.parallelize(sc.textFile(data_path) \
          .map(lambda line: split_line(line)) \
          .map(lambda (words, label): (words_to_ids(words, broadcast_word2id.value, max_sentence_length), label)) \
          .map(lambda (ids, label): to_sample(ids, broadcast_drop.value, label)).take(128 * 5))
        # print(test_rdd.take(2))
        # a = test_rdd.first()
        # # print(type(a))
        print(test_rdd.count())
        
        # init bigdl
        # redire_spark_logs()
        # show_bigdl_info_logs()
        # init_engine()
        
        # # load model to bigdl
        # model_def = model_path + "/model.pb"
        # model_variable = model_path + "/model.bin"
        # inputs = ["input_x", "dropout_keep_prob"]
        # outputs = ["textCnn/output/probs"]
        # model = Model.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float", bin_file=model_variable)

        folder = "/home/yang/sources/zoo/pyzoo/analytics-zoo-rnn"
        model = TFNet.from_export_folder(folder)
        
        # evaluate model
        print("Starting evaluation")
        start = time()
        results = model.evaluate(test_rdd, batch_size, [Top1Accuracy()])
        stop = time()
        print ("Evaluation finished. Cost " + str(stop - start) + " seconds.")
        
        for result in results:
            print(result)

    elif options.action == "prediction":
        sqlContext = SQLContext(sc)

        # load word2id
        sc.addFile(word2id_path)
        word2id = load_word2id("word2id")
        
        # create broadcast variable
        drop_percent = 1.0
        drop = JTensor.from_ndarray(np.full([300], drop_percent))
        broadcast_drop = sc.broadcast(drop)
        broadcast_word2id = sc.broadcast(word2id)

        # create test_rdd
        test_dataset = sc.textFile(data_path).map(lambda line: line.split("\t")) \
            .filter(lambda columns: len(columns) >= 3)
        print(test_dataset.first())
        test_rdd = test_dataset.map(lambda columns: \
            to_predict_sample(columns[2], broadcast_word2id.value, max_sentence_length, broadcast_drop.value))
        print(test_rdd.count())
        
        # init bigdl
        # redire_spark_logs()
        # show_bigdl_info_logs()
        # init_engine()
        
        # # load model to bigdl
        # model_def = model_path + "/model.pb"
        # model_variable = model_path + "/model.bin"
        # inputs = ["input_x", "dropout_keep_prob"]
        # outputs = ["textCnn/output/probs"]
        # model = Model.load_tensorflow(model_def, inputs, outputs, byte_order = "little_endian", bigdl_type="float", bin_file=model_variable)
        #
        folder = "/home/yang/sources/zoo/pyzoo/analytics-zoo-rnn"
        model = TFNet.from_export_folder(folder)

        # evaluate model
        print("Starting prediction")
        start = time()
        results = model.predict_distributed(test_rdd, batch_size).map(lambda result: np.argmax(result))
        test_dataset.zip(results).map(lambda (columns, label): columns[0] + "\t" + columns[1] + "\t" + str(label)).saveAsTextFile(output_path)
        # results = results.zip(test_dataset)
        stop = time()
        print ("Evaluation finished. Cost " + str(stop - start) + " seconds.")


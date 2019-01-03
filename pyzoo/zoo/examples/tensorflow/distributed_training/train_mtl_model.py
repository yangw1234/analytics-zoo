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
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pickle
from os import path
import tensorflow as tf

from tensorflow.python.keras.utils import to_categorical

from nlp_architect.contrib.tensorflow.python.keras.callbacks import ConllCallback
from nlp_architect.data.intent_datasets import SNIPS
from nlp_architect.models.intent_extraction import MultiTaskIntentModel
from nlp_architect.utils.embedding import get_embedding_matrix, load_word_embeddings
from nlp_architect.utils.generic import one_hot
from nlp_architect.utils.io import validate, validate_existing_directory, \
    validate_existing_filepath, validate_parent_exists
from nlp_architect.utils.metrics import get_conll_scores

from bigdl.optim.optimizer import MaxEpoch
from zoo.common.nncontext import init_nncontext
from zoo.pipeline.api.net import TFOptimizer, TFDataset


def validate_input_args():
    global model_path
    validate((args.b, int, 1, 100000000))
    validate((args.e, int, 1, 100000000))
    validate((args.sentence_length, int, 1, 10000))
    validate((args.token_emb_size, int, 1, 10000))
    validate((args.intent_hidden_size, int, 1, 10000))
    validate((args.lstm_hidden_size, int, 1, 10000))
    validate((args.tagger_dropout, float, 0, 1))
    model_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_path))
    validate_parent_exists(model_path)
    model_info_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_info_path))
    validate_parent_exists(model_info_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--dataset_path', type=validate_existing_directory,
                        default="/home/yang/sources/datasets/" +
                                "nlu-benchmark/2017-06-custom-intent-engines",
                        help='dataset directory')
    parser.add_argument('--sentence_length', type=int, default=30,
                        help='Max sentence length')
    parser.add_argument('--token_emb_size', type=int, default=100,
                        help='Token features embedding vector size')
    parser.add_argument('--intent_hidden_size', type=int, default=100,
                        help='Intent detection LSTM hidden size')
    parser.add_argument('--lstm_hidden_size', type=int, default=150,
                        help='Slot tags LSTM hidden size')
    parser.add_argument('--tagger_dropout', type=float, default=0.5,
                        help='Slot tags dropout value')
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to word embedding model file')
    parser.add_argument('--use_cudnn', default=False, action='store_true',
                        help='use CUDNN based LSTM cells')
    parser.add_argument('--model_path', type=str, default='model.h5',
                        help='Model file path')
    parser.add_argument('--model_info_path', type=str, default='model_info.dat',
                        help='Path for saving model topology')
    args = parser.parse_args()
    validate_input_args()

    # load dataset
    print('Loading dataset')
    dataset = SNIPS(path=args.dataset_path,
                    sentence_length=args.sentence_length)

    train_x, train_char, train_i, train_y = dataset.train_set
    test_x, test_char, test_i, test_y = dataset.test_set

    test_y = to_categorical(test_y, dataset.label_vocab_size)
    train_y = to_categorical(train_y, dataset.label_vocab_size)
    train_i = one_hot(train_i, len(dataset.intents_vocab))
    test_i = one_hot(test_i, len(dataset.intents_vocab))

    train_inputs = [train_x, train_char]
    train_outs = [train_i, train_y]
    test_inputs = [test_x, test_char]
    test_outs = [test_i, test_y]

    sc = init_nncontext()

    train_list = []
    for i in range(0, len(train_x)):
        train_list.append([train_x[i], train_char[i], train_i[i], train_y[i]])

    train_rdd = sc.parallelize(train_list)

    print('Building model')
    from tensorflow.python.ops import variable_scope

    def variable_creator(**kwargs):
        kwargs["use_resource"] = False
        return variable_scope.default_variable_creator(None, **kwargs)
    getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
    with variable_scope.variable_creator_scope(getter):
        model = MultiTaskIntentModel(use_cudnn=args.use_cudnn)
        model.build(dataset.word_len,
                    dataset.label_vocab_size,
                    dataset.intent_size,
                    dataset.word_vocab_size,
                    dataset.char_vocab_size,
                    word_emb_dims=args.token_emb_size,
                    tagger_lstm_dims=args.lstm_hidden_size,
                    dropout=args.tagger_dropout)

    dataset = TFDataset.from_rdd(train_rdd,
                                 names=["i1", "i2", "o1", "o2"],
                                 shapes=[[30], [30, 12], [7], [30, 73]],
                                 types=[tf.int32, tf.int32, tf.float32, tf.float32],
                                 batch_size=12)

    model = model.model
    from tensorflow.keras.optimizers import Adam
    model.optimizer = Adam()
    model.metrics = []

    optimizer = TFOptimizer.from_keras(model, dataset)
    optimizer.optimize(end_trigger=MaxEpoch(10))

    # initialize word embedding if external model selected
    if args.embedding_model is not None:
        print('Loading external word embedding')
        embedding_model, _ = load_word_embeddings(args.embedding_model)
        embedding_mat = get_embedding_matrix(embedding_model, dataset.word_vocab)
        model.load_embedding_weights(embedding_mat)

    conll_cb = ConllCallback(test_inputs, test_y, dataset.tags_vocab.vocab, batch_size=args.b)
    # train model
    model.fit(x=train_inputs, y=train_outs,
              batch_size=args.b, epochs=args.e,
              validation=(test_inputs, test_outs),
              callbacks=[conll_cb])
    print('Training done')

    print('Saving model')
    model.save(args.model_path)
    with open(args.model_info_path, 'wb') as fp:
        info = {
            'type': 'mtl',
            'tags_vocab': dataset.tags_vocab.vocab,
            'word_vocab': dataset.word_vocab.vocab,
            'char_vocab': dataset.char_vocab.vocab,
            'intent_vocab': dataset.intents_vocab.vocab,
        }
        pickle.dump(info, fp)

    # test performance
    predictions = model.predict(test_inputs, batch_size=args.b)
    eval = get_conll_scores(predictions, test_y,
                            {v: k for k, v in dataset.tags_vocab.vocab.items()})
    print(eval)

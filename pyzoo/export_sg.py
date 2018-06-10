import tensorflow as tf

with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
    new_saver = tf.train.import_meta_graph("/home/yang/sources/datasets/sougou/textcnn/model-7000.meta")
    new_saver.restore(sess, "/home/yang/sources/datasets/sougou/textcnn/model-7000")

    # tf.summary.FileWriter("./graph", sess.graph)

    input_x = tf.get_default_graph().get_tensor_by_name("shuffle_batch:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("dropout_keep_prob:0")
    output_tensor = tf.get_default_graph().get_tensor_by_name("textCnn/output/probs:0")

    from zoo.util.tf import export_tf
    export_tf(sess, "./analytics-zoo-rnn", [input_x, keep_prob], [output_tensor])
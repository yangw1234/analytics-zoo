${SPARK_HOME}/bin/spark-submit \
    --master local[*] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.examples.nnframes.finetune.TransferLearning \
    dist/lib/analytics-zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar

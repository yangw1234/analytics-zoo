master=spark://Gondolin-Node-050:7077
master=local[16]

${SPARK_HOME}/bin/spark-submit \
--verbose \
--master $master \
--conf spark.executor.cores=16 \
--total-executor-cores 16 \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.tfnet.Predict \
dist/lib/analytics-zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar --partition 16


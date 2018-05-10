export ZOO_HOME=~/sources/zoo/dist
${ZOO_HOME}/scripts/jupyter-with-zoo.sh \
    --master local[4] \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g

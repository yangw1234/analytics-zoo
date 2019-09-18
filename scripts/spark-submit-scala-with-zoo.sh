#!/bin/bash

# Check environment variables
if [[ -z "${ANALYTICS_ZOO_HOME}" ]]; then
    echo "Please set ANALYTICS_ZOO_HOME environment variable"
    exit 1
fi

if [[ -z "${SPARK_HOME}" ]]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

export SPARK_CMD=spark-submit

bash ${ANALYTICS_ZOO_HOME}/bin/analytics-zoo-base.sh \
    $*
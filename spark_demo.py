#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:20:31 2021

@author: elie
"""

import time
import pyspark

sc = pyspark.SparkContext.getOrCreate('local[*]')

#txt = sc.textFile('file:////usr/share/doc/python/copyright')
#print(txt.count())

#python_lines = txt.filter(lambda line: 'python' in line.lower())
#print(python_lines.count())


big_list = list(range(5000000))

start_time_spark = time.time()

# instantiate rdd working on 2 logical CPU?
rdd = sc.parallelize(big_list)

# just keep the odd number
odds_spark = rdd.filter(lambda x: x % 2 != 0)

# no computation is done until you request with take
odds_spark = odds_spark.collect()

# how long did it take? 
print("--- %s seconds --- SPARK" % (time.time() - start_time_spark))


# let's compare speed of execution between good'old Python and Spark
start_time_python = time.time()
odds_python = list(filter(lambda x: x % 2 != 0, big_list))
print("--- %s seconds --- Python" % (time.time() - start_time_python))

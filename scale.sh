#!/bin/bash
# based on the number files, increaese replicas

pushd algorithm/data/
files_count=$(ls -1 input-data-on-laptop | wc -l)
kubectl scale sts processor --replicas $(echo $files_count)


COUNTER=0
for file in $(ls -1 input-data-on-laptop); 
do 
  mkdir -p input-data-inside-pod/processor-$COUNTER/
  cp input-data-on-laptop/$file input-data-inside-pod/processor-$COUNTER/
  COUNTER=$[$COUNTER +1]
done
popd

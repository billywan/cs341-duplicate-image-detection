#!/bin/bash
a=0
for i in *_*; do
  new=$(printf "data_batch_%03d" "$a") #04 pad to length of 4
  ln -- "$i" "/Users/EricX/Desktop/CS341/test_folder/$new" #/mnt/data2/data_batches_01_total
  let a=a+1
done
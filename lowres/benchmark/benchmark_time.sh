#!/bin/bash
num_threads=$1
echo "Choose the model name from the given list: deepmedic39, unet3d, mobilenetv2unet3d, enet3d, lowres"

while read -p "model name: " model_name; do
  case $model_name in
  deepmedic39)
    ps=90
    break
    ;;
  unet3d)
    ps=64
    break
    ;;
  mobilenetv2unet3d)
    ps=96
    break
    ;;
  enet3d)
    ps=96
    break
    ;;
  lowres)
    ps=96
    break
    ;;
  *) echo "Unexpected model name \"$model_name\" given. Please re-enter: " ;;
  esac
done

export OMP_NUM_THREADS=$num_threads
python model_predict.py --model_name $model_name --ps $ps

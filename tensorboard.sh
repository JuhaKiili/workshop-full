docker run --rm -it -t -v $PWD:/work -p 8888:8888 juhavalohai/workshop:$1 tensorboard --logdir=/work/log  --port=8888

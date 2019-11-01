sh /work/tensorboard.sh &
python -W ignore /work/train.py \
    -epochs=50 \
    -batch_size=16 \
    -image_size=128 \
    -learning_rate=0.001 \
    -drop_out=0.8 \
    -filter_count=16 \
    -images_count=1000000 \
    -validation_count=250 \
    -name=testo1

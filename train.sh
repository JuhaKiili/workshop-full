sh /work/tensorboard.sh &
python -W ignore /work/train.py \
    -epochs=1000 \
    -batch_size=100 \
    -image_size=128 \
    -dense_size=512 \
    -learning_rate=0.001 \
    -drop_out=0.8 \
    -filter_count=32 \
    -images_count=1000000 \
    -validation_count=1000 \
    -rotation=20 \
    -shear=0.1 \
    -zoom=0.2 \
    -shift=0.1 \
    -fill_mode=reflect

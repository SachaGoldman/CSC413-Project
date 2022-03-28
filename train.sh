python ./train.py \
    --content_dir ./datasets/cocotrain2014_small \
    --style_dir ./datasets/wikiart_small \
    --vgg ./weights/vgg_normalised.pth \
    --max_iter 1000 \
    --batch_size 8 \
    --n_threads 4 \
    --save_model_interval 500 \
    --dino_encoder none
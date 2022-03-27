python ./train.py \
    --content_dir ./datasets/coco \
    --style_dir ./datasets/wikiart \
    --vgg ./weights/vgg_normalised.pth \
    --batch_size 8 \
    --n_threads 4 \
    --save_model_interval 10000 \
    --dino_encoder none
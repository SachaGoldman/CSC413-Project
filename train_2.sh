VENV_PATH="venv/bin/activate"

if [ -d $VENV_PATH ] && echo $VENV_PATH
then
    source $VENV_PATH
fi

python ./train.py \
    --content_dir ./datasets/cocotrain2014 \
    --style_dir ./datasets/wikiart \
    --vgg ./weights/vgg_normalised.pth \
    --save_dir ./experiment2 \
    --log_dir ./logs2 \
    --lr_decay 2.5e-6 \
    --max_iter 640000 \
    --batch_size 2 \
    --gradient_accum_steps 4 \
    --n_threads 4 \
    --save_model_interval 50000 \
    --dino_encoder vits8 \
    --amp
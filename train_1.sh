VENV_PATH="venv/bin/activate"

if [ -d $VENV_PATH ] && echo $VENV_PATH
then
    source $VENV_PATH
fi

python ./train.py \
    --content_dir ./datasets/cocotrain2014 \
    --style_dir ./datasets/wikiart \
    --vgg ./weights/vgg_normalised.pth \
    --save_dir ./experiment1 \
    --log_dir ./logs1 \
    --lr_decay 5e-6 \
    --max_iter 320000 \
    --batch_size 4 \
    --gradient_accum_steps 2 \
    --n_threads 4 \
    --save_model_interval 50000 \
    --dino_encoder vits8 \
    --freeze_encoder_s \
    --freeze_encoder_c \
    --amp
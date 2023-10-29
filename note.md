#### inpaint
conda activate sd
python inpaint.py


#### render mask
conda activate guassian_splatting
CUDA_VISIBLE_DEVICES=1 python render.py -m output/image_sparse --skip_train --iteration 7000
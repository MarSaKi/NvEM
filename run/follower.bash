name=NvEM
flag="--attn soft --train listener
      --featdropout 0.3
      --visual_feat --angle_feat
      --feedback sample
      --mlWeight 0.2
      --features places365
      --glove_dim 300 --top_N_obj 8
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
name=default
flag="--attn soft --train validlistener
      --featdropout 0.3
      --load snap/NvEM_bt/state_dict/best_val_unseen
      --visual_feat --angle_feat
      --features places365
      --glove_dim 300 --top_N_obj 8 --submit
      --subout max --dropout 0.5 --maxAction 25"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
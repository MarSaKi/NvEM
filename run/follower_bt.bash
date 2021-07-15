name=NvEM_bt
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker/state_dict/best_val_unseen_bleu
      --load snap/NvEM/state_dict/best_val_unseen
      --features places365

      --visual_feat --angle_feat
      --glove_dim 300 --top_N_obj 8

      --accumulateGrad
      --featdropout 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
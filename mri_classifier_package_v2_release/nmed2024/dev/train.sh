#!/bin/bash -l

# run this script from adrd_tool/

source /mnt/d/hahmed/py_envs/nmed/bin/activate
#pip install .

# install the package
# cd adrd_tool
# pip install -e .

# define the variables
prefix=""
dat_file='/mnt/d/NACC/converted_data-all-aforms-b1b2b8-c1c2c2t-apoe-cdr.csv' # the test data path

data_path="${prefix}/mnt/d/NACC/test_converted_data-5000-aforms-b1b2b8-c1c2c2t-apoe-cdr.csv"
train_path="${prefix}/mnt/d/NACC/retrain_train_aforms-b1b2b8-c1c2c2t-apoe-cdr.csv"
vld_path="${prefix}/mnt/d/NACC/retrain_vld_aforms-b1b2b8-c1c2c2t-apoe-cdr.csv"
test_path="${prefix}/mnt/d/NACC/retrain_test_aforms-b1b2b8-c1c2c2t-apoe-cdr.csv"
cnf_file="${prefix}/mnt/d/hahmed/nmed2024/dev/data/toml_files/default_conf_modified.toml"
imgnet_ckpt="${prefix}/mnt/d/hahmed/nmed2024/dev/ssl_mri/pretrained_models/model_swinvit.pt"

# Note for setting the flags
# 1. If training without MRIs
# img_net="NonImg"
# img_mode = -1
# 2. if training with MRIs
# img_net: [ViTAutoEnc, DenseNet, SwinUNETR]
# img_mode = 0
# 3. if training with MRI embeddings
# img_net: [ViTEmb, DenseNetEMB, SwinUNETREMB, NonImg]
# img_mode = 1


img_net="NonImg"
img_mode=-1


#img_net="SwinUNETREMB"
#img_mode=1

# img_net="DenseNet"
# img_mode=0

ckpt_path="/mnt/d/hahmed/nmed2024/dev/ssl_mri/pretrained_models/ckpt_swinunetr_stripped_MNI.pt"

# run train.py 
python /mnt/d/hahmed/nmed2024/dev/train.py --data_path $data_path --train_path $train_path --vld_path $vld_path --test_path $test_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 256 --nhead 1 \
                    --num_epochs 256 --batch_size 128 --lr 0.001 --gamma 0 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)" --imgnet_ckpt ${imgnet_ckpt} \
                    --patch_size 16 --ckpt_path $ckpt_path --cnf_file ${cnf_file} --train_path ${train_path} --vld_path ${vld_path} --data_path ${data_path}  \
                    --fusion_stage middle --imgnet_layers 4 --weight_decay 0.0005 --ranking_loss --save_intermediate_ckpts --load_from_ckpt #--train_imgnet #--balanced_sampling
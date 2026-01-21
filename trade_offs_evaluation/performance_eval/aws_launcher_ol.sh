#Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#SPDX-License-Identifier: Apache-2.0



# Set the basic path for crypten
export CRYPTEN_PATH=/home/$USER/sok-cpcl/trade_offs_evaluation/Crypten
export CUSTOM_PATH=/home/$USER/sok-cpcl/trade_offs_evaluation/performance_eval
export SRC_PATH=/home/$USER/sok-cpcl/trade_offs_evaluation/utils



python $CRYPTEN_PATH/scripts/aws_launcher.py \
    --ssh_key_file=/Users/$USER/path_to_the_keys/aws_keys.pem\
    --instances=i-01234567890123ab,i-09876543219876ab\
    --regions=eu-central-1\
    --master_port=5000 \
    --aux_files $SRC_PATH/models.py,$SRC_PATH/mpc_dpsgd_trainer.py,$SRC_PATH/eval_utils.py,$CUSTOM_PATH/ot_evaluation.py,$CUSTOM_PATH/training.py,$CUSTOM_PATH/data/eval_config.yaml\
    $CUSTOM_PATH/ol_launcher.py \
    --epochs 1\
    --examples 500 \
    --batch_size 500 \
    --lr 0.1 \
    --epsilon 1 \
    --clip_threshold 4.0 \
    --n_iterations 10 \
    --num_labels 10 \
    --noise_type local \
    --world_size 2 \
    --device cpu \
    --model nn 


# For WAN need to specify multiple regions and one key file per region
#python $CRYPTEN_PATH/scripts/aws_launcher.py \
#    --ssh_key_file=/Users/$USER/path_to_the_keys/aws_keys_region_1.pem,/Users/$USER/path_to_the_keys/aws_keys_region_2.pem \
#    --instances=i-01234567890123ab,i-09876543219876ab\
#    --regions=eu-central-1,eu-west-2 \
#    --master_port=5000 \
#    --aux_files $SRC_PATH/models.py,$SRC_PATH/mpc_dpsgd_trainer.py,$SRC_PATH/eval_utils.py,$CUSTOM_PATH/ot_evaluation.py,$CUSTOM_PATH/training.py,$CUSTOM_PATH/data/eval_config.yaml\
#    $CUSTOM_PATH/ol_launcher.py 

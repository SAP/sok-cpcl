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

# --data_path
# --download_data

python $CRYPTEN_PATH/scripts/aws_launcher.py \
    --ssh_key_file=/Users/$USER/aws_utils/aws_keys.pem\
    --instances=i-0f3edff27167c4089,i-05de0c92c3bff5aa6\
    --regions=eu-central-1\
    --master_port=5000 \
    --aux_files $SRC_PATH/models.py,$SRC_PATH/mpc_dpsgd_trainer.py,$SRC_PATH/eval_utils.py,$CUSTOM_PATH/ot_mpc_noise.py,$CUSTOM_PATH/training.py,$CUSTOM_PATH/data/eval_config.yaml\
    $CUSTOM_PATH/mpc_noise_launcher.py --noise_type local

# noise_type local or global sample local or global gaussian noise
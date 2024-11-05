if [ "$1" = "" ]; then
    config_name=train_default.yaml
else
    config_name=$1
fi

echo running train.py with config $config_name, pretraining...
accelerate launch train.py --pretrain --config $config_name
echo running train.py with config $config_name, training...
accelerate launch train.py --train --config $config_name

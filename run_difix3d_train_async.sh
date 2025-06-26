# Launch training in the background without the web viewer.
# Logs can be inspected with TensorBoard:  tensorboard --logdir ./outputs
./run_difix3d_train.sh > train.log 2>&1 &


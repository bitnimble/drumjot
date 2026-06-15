#!/usr/bin/env bash
# Create + start the drumjot sandbox container.
#
# --shm-size=2g: the default 64MB /dev/shm makes PyTorch DataLoader workers
# (num_workers>0) hang/OOM when they share tensors; 2g gives prefetch room.
# shm size is fixed at creation, so changing it means recreating the container:
#   sudo docker stop drumjot-sandbox && sudo docker rm drumjot-sandbox && sandbox/start.sh
sudo docker run -d --name drumjot-sandbox --gpus all --shm-size=2g \
  -v /capsule/config/docker-volume/codebox/home/bitnimble/code/drumjot:/home/bitnimble/code/drumjot \
  -v /codebox-workspace:/codebox-workspace \
  drumjot-sandbox

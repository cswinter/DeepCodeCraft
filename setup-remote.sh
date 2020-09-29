#!/bin/bash

set -euxo pipefail


PORT="$1"
INSTANCE="$2"

rsync -azP -e "ssh -p $PORT" . root@ssh$INSTANCE.vast.ai:src/DeepCodeCraft
rsync -azP -e "ssh -p $PORT" ../CodeCraftServer/ root@ssh$INSTANCE.vast.ai:src/CodeCraftServer
rsync -azP -e "ssh -p $PORT" /home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models/standard/noble-sky-145M.pt root@ssh$INSTANCE.vast.ai:/home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models/standard/
rsync -azP -e "ssh -p $PORT" /home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models/standard/radiant-sun-35M.pt root@ssh$INSTANCE.vast.ai:/home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models/standard/

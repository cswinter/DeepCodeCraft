#!/bin/bash

python showmatch.py ~/Dropbox/artifacts/DeepCodeCraft/golden-models/arena_medium/{honest-field-109M,true-valley-150M}.pt --hardness=1 --task=ARENA_MEDIUM --num_envs=8

# python showmatch.py C:\Users\cleme\Dropbox\artifacts\DeepCodeCraft\golden-models\arena_medium\honest-field-109M.pt C:\Users\cleme\Dropbox\artifacts\DeepCodeCraft\golden-models\arena_medium\true-valley-150M.pt --hardness=1 --task=ARENA_MEDIUM --num_envs=8

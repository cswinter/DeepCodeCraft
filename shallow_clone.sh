#!/bin/bash

set -euxo pipefail

REMOTE="$1"
REVISION="$2"
TARGET_DIR="$3"

mkdir -p $TARGET_DIR
git -C $TARGET_DIR init
git -C $TARGET_DIR remote add origin $REMOTE
git -C $TARGET_DIR fetch --depth 1 origin $REVISION
git -C $TARGET_DIR checkout FETCH_HEAD
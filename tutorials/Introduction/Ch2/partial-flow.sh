#!/bin/bash

PATH_TO_COMP="tutorials/Introduction/Ch2/out/comp"

# Export to DOT (visual)
./bin/export-dot \
  $PATH_TO_COMP/handshake_export.mlir \
  --mode=visual \
  --edge-style=ortho \
  --timing-models=./data/components.json \
  > $PATH_TO_COMP/visual.dot

# Convert to PNG (visual)
dot -Tpng $PATH_TO_COMP/visual.dot > $PATH_TO_COMP/visual.png

# Export to DOT (legacy)
./bin/export-dot \
  $PATH_TO_COMP/handshake_export.mlir \
  --mode=legacy \
  --edge-style=ortho \
  --timing-models=./data/components.json \
  > $PATH_TO_COMP/loop_store.dot

# Convert to PNG (visual)
dot -Tpng $PATH_TO_COMP/loop_store.dot > $PATH_TO_COMP/loop_store.png

# Simulate using the frontend
./bin/dynamatic --run tutorials/Introduction/Ch2/loop-store-partial.dyn

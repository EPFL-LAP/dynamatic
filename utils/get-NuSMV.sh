#!/usr/bin/env bash

# The equivalence checking is based on NuSMV (https://nusmv.fbk.eu/).
# The official version on supports printing 2^16 state spaces. 
# To circumvent this problem a modified binary supporting 2^24 states can be downloaded.

mkdir ext
LOCATION=ext/NuSMV

wget https://github.com/ETHZ-DYNAMO/NuSMV-2.7.0/releases/download/v1.0.0/NuSMV -O $LOCATION

chmod +x $LOCATION
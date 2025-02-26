#!/bin/bash

#for MLD_method in RHO FIXED500m; do
for MLD_method in RHO ; do
    python3 postprocess_ECCO.py --MLD-method $MLD_method --nproc 10
done
    
wait



# No need to remap now
#./postprocess_remap_ECCO.sh

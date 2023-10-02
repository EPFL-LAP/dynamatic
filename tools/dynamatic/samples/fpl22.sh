# Run all integration tests available on the repository

# Indicate the path to your legacy Dynamatic install here (required for write-hdl)
set-legacy-path   ../dynamatic-utils/legacy-dynamatic/dhls/etc/dynamatic

# bicg
set-src           integration-test/src/bicg/bicg.c
synthesize        --simple-buffers
write-hdl
simulate

# binary_search
set-src           integration-test/src/binary_search/binary_search.c
synthesize        --simple-buffers
write-hdl
simulate

# fir
set-src           integration-test/src/fir/fir.c
synthesize        --simple-buffers
write-hdl
simulate

# gaussian
set-src           integration-test/src/gaussian/gaussian.c
synthesize        --simple-buffers
write-hdl
simulate

# gcd
set-src           integration-test/src/gcd/gcd.c
synthesize        --simple-buffers
write-hdl
simulate

# matvec
set-src           integration-test/src/matvec/matvec.c
synthesize        --simple-buffers
write-hdl
simulate

# polyn_mult
set-src           integration-test/src/polyn_mult/polyn_mult.c
synthesize        --simple-buffers
write-hdl
simulate

# sobel
set-src           integration-test/src/sobel/sobel.c
synthesize        --simple-buffers
write-hdl
simulate

# stencil_2d
set-src           integration-test/src/stencil_2d/stencil_2d.c
synthesize        --simple-buffers
write-hdl
simulate

# Exit the frontend
exit 
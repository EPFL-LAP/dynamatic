# Fully run (from C to synthesis) all integration tests available on the 
# repository

# Indicate the path to Dynamatic's top-level directory here (leave unchanged if
# running the frontend from the top-level directory)
set-dynamatic-path  .

set-src           integration-test/bicg/bicg.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/binary_search/binary_search.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/fir/fir.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/gaussian/gaussian.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/gcd/gcd.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/gemver/gemver.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/if_loop_1/if_loop_1.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/if_loop_2/if_loop_2.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/if_loop_3/if_loop_3.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/iir/iir.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/image_resize/image_resize.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/insertion_sort/insertion_sort.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/kernel_2mm/kernel_2mm.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/kernel_3mm/kernel_3mm.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/loop_array/loop_array.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/matrix/matrix.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/matrix_power/matrix_power.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/matvec/matvec.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/mul_example/mul_example.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/pivot/pivot.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/polyn_mult/polyn_mult.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/simple_example/simple_example.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/sobel/sobel.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/stencil_2d/stencil_2d.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/sumi3_mem/sumi3_mem.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/threshold/threshold.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/triangular/triangular.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/vector_rescale/vector_rescale.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/video_filter/video_filter.c
compile
write-hdl
simulate
synthesize

# Memory-oriented tests
set-src           integration-test/memory/test_memory_1/test_memory_1.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_2/test_memory_2.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_3/test_memory_3.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_4/test_memory_4.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_5/test_memory_5.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_6/test_memory_6.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_7/test_memory_7.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_8/test_memory_8.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_9/test_memory_9.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_10/test_memory_10.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_11/test_memory_11.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_12/test_memory_12.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_13/test_memory_13.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_14/test_memory_14.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_15/test_memory_15.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_16/test_memory_16.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_17/test_memory_17.c
compile
write-hdl
simulate
synthesize
set-src           integration-test/memory/test_memory_18/test_memory_18.c
compile
write-hdl
simulate
synthesize

# Exit the frontend
exit 
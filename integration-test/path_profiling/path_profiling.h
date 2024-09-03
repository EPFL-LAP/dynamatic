#ifndef PATH_PROFILING_PATH_PROFILING_H
#define PATH_PROFILING_PATH_PROFILING_H
typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

#define LOOP_BOUND 20

void path_profiling(out_int_t a[LOOP_BOUND], out_int_t b[LOOP_BOUND],
                    in_int_t var[LOOP_BOUND]);
#endif // PATH_PROFILING_PATH_PROFILING_H

#ifndef TEST_MEMORY_DEPS_TEST_MEMORY_DEPS_H
#define TEST_MEMORY_DEPS_TEST_MEMORY_DEPS_H

typedef int in_int_t;
typedef int inout_int_t;

void test_memory_deps(in_int_t load_addrs[1000], in_int_t store_addrs[1000], inout_int_t data[1000], in_int_t n);

#endif // TEST_MEMORY_DEPS_TEST_MEMORY_DEPS_H

# Validating Dynamatic Using a Random Program Generator

Dynamatic has a C-based random program generator (i.e., a "fuzzer") to detect
miscompilation.
This document provides instruction for using and extending the fuzzer.

## Using HLSFuzzer to Find Functional Bugs

HLSFuzzer can perform differential test against Clang. To generate random C
code to test Dynamatic

```
build/bin/hls-fuzzer --functional --target random-c
```

**Number of threads**

```
build/bin/hls-fuzzer -j <num-threads>
```

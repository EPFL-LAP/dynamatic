# Integration Tests

This document describes the setup used for running integration tests in Dynamatic, based on [GoogleTest](https://google.github.io/googletest/), [Ninja](https://ninja-build.org/)
and [CMake's ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

## Introduction

In order to avoid confusion, we introduce the following terminology.

A **benchmark** is a piece of .c code, commonly called a *kernel*, which is written in order to be compiled by Dynamatic into a HDL representation of a dataflow circuit. Confusingly, benchmarks are located in the `integration-test` folder. We also say that benchmarks are *testing resources*, since they are files that are loaded and used when running a test.

An **integration test** is a piece of code which runs a Dynamatic in order to compile, convert into HDL and simulate a certain benchmark, with some specific parameters. To clarify the difference, `integration-test/fir/fir.c` is a benchmark, while a C++ function which runs Dynamatic `set-src <benchmark>`, `compile`, `write-hdl` and `simulate` commands is an integration test. Note that integration tests differ by parameters used; a test that runs `compile --buffer-algorithm on-merges` is not the same as a test that runs `compile --buffer-algorithm fpga20`, even if they use the same benchmark.



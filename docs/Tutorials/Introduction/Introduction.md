# Introduction to Dynamatic

This tutorial is meant as the entry-point for new Dynamatic users and will guide you through your first interactions with the compiler and its surrounding toolchain. Following it requires that you have Dynamatic built locally on your machine, either from source or using our custom virtual machine ([VM setup instructions](../../VMSetup.md)).

> [!WARNING]
> Note that the virtual machine does not contain an MILP solver; when using frontend scripts, you will have to provide the `--simple-buffers` flag to the `compile` command to instruct it to not rely on an MILP solver for buffer placement). Unfortunately, this will affect the circuits you generate as part of the exercises and you may therefore obtain different results from what the tutorial describes.

It is divided in the following two chapters.

- [Chapter #1 - Using Dynamatic](UsingDynamatic.md) | We use Dynamatic's frontend to synthesize our first dataflow circuit from C code, then visualize it using our interactive dataflow visualizer.
- [Chapter #2 - Modifying Dynamatic](ModifyingDynamatic.md) | We write a small compiler transformation pass in C++ to try to improve circuit performance and decrease area, then debug it using the visualizer.
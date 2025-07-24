# Dynamatic

Dynamatic is an academic, open-source high-level synthesis compiler that
produces synchronous dynamically-scheduled circuits from C/C++ code. Dynamatic
generates synthesizable RTL which currently targets Xilinx FPGAs and delivers
significant performance improvements compared to state-of-the-art commercial
HLS tools in specific situations (e.g., applications with irregular memory
accesses or control-dominated code). The fully automated compilation flow of
Dynamatic is based on MLIR. It is customizable and extensible to target
different hardware platforms and easy to use with commercial tools such as
Vivado (Xilinx) and Modelsim (Mentor Graphics).

We welcome contributions and feedback from the community. If you would like to
participate, please check out our [contribution
guidelines](./DeveloperGuide/IntroductoryMaterial/Contributing.md)

## Using Dynamatic

To get started using Dynamatic (after setting it up), check out our
[introductory tutorial](./GettingStarted/Tutorials/Introduction/Introduction.md),
which guides you through your first compilation of C code into a synthesizable
dataflow circuit! If you want to start modifying Dynamatic and are new to MLIR
or compilers in general, our [MLIR primer](./DeveloperGuide/CompilerIntrinsics/MLIRPrimer.md)
and [pass creation tutorial](./DeveloperGuide/IntroductoryMaterial/Tutorials/CreatingPasses/CreatingPassesTutorial.md)
will help you take your first steps.

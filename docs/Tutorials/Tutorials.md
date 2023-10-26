# Tutorials

Welcome to the Dynamatic tutorials!

To encourage contributions to the project, we aim to support newcomers to the worlds of software development and compilers by providing development tutorials that can help them take their first steps inside the codebase. They are mostly aimed at people who have no or little compiler development experience, especially with the MLIR compiler infrastructure with which Dynamatic is deeply intertwined. Some prior knowledge of C++ (more generally, of object-oriented programming) and of the theory behind dataflow circuits is assumed.

## [The MLIR Primer](MLIRPrimer.md)

This tutorial, heavily based on [MLIR's official language reference](https://mlir.llvm.org/docs/LangRef/), is meant as a quick introduction to MLIR and its core constructs. C++ code snippets are peperred through the tutorial in an attempt to ease newcomers to the framework's C++ API and provide some initial code guidance. 

## [Creating compiler passes](CreatingPasses/CreatingPasses.md)

This tutorial goes through the creation of a simple compiler transformation pass that operates on *Handshake*-level IR (i.e., on dataflow circuits modeled in MLIR). It goes into details into all the code that one needs to write to declare a pass in the codebase, implement it, and then run it on some input code using the `dynamatic-opt` tool. It then touches on different ways to write the same pass as to give an idea of MLIR's code transformation capabilities.  

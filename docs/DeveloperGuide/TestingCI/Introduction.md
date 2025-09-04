# Introduction

This document describes the basic high-level concepts behind integration testing and GitHub Actions.

## Integration testing

As per [Wikipedia's definition](https://en.wikipedia.org/wiki/Integration_testing), integration testing is a form of software testing in which multiple software components, modules, or services are tested together to verify they work as expected when combined. The focus is on testing the interactions and data exchange between integrated parts, rather than testing components in isolation. 

In the case of Dynamatic, these modules are the different parts of the HLS flow: the compiler frontend, all of the intermediate MLIR transformations, compiler backend (i.e. conversion to HDL form) and HDL simulation. In simpler terms, the goal of integration testing is: **"Make sure it is possible (without errors/crashes) to take a single valid C source file, run it all the way through Dynamatic, which results in a HDL file, and that simulation of said HDL behaves correctly, i.e. its behavior matches what the original C source describes."** Even more simply, make sure that Dynamatic as a whole works as it should.

Different integration tests run the program in different ways. For example, one integration test may run Dynamatic to compile some file named `binary_search.c`, using the FPGA'20 buffering algorithm and the VHDL backend. Another integration test may compile `fir.c`, with on-merges buffering and the Verilog backend. It is immediately clear that the number of possible integration tests is very large; it is infeasible to test all possible combinations. For this reason, Dynamatic has a set of integration tests that are known to pass when run on main branch Dynamatic. Of course, this set is fairly small compared to the set of all possible tests, however, it provides a baseline for assuring that broken code does not end up in main.

Additionally, Dynamatic's integration tests also measure the performance of the resulting HDL circuits.

For more technical details on how integration tests are implemented in Dynamatic, see [Integration tests](IntegrationTests.md).

### Integration testing policy

Developers should keep some rules in mind regarding the use of integration tests and their purpose. For the sake of clarity, let *S* denote the set of integration tests that is present in the main branch at the current time.

0. Test regularly to catch problems as soon as they appear.

1. Before merging your branch, all tests in *S* must pass. That is, you must not merge code which breaks some integration tests into main.

2. Your branch's features should not result in Dynamatic's performance (in terms of resulting circuit performance) being degraded. That is, if an integration test has a measured circuit performance of *N* cycles in your branch, and *M* cycles in main, you should have *N≤M*, for all integration tests in *S*. However, if there is a good reason for a performance degradation (e.g. it is only 2% slower on a single test, but it is 10% faster on all of the others), please explain it in your pull request.

3. You are allowed (and encouraged) to add new integration tests along with your feature, in the same branch (of course, along with an explanation in the pull request). For example, imagine that the Verilog backend is not tested as part of *S* because it does not work properly. Then, your branch adds a feature that fixes the Verilog backend. In this case, you absolutely should add integration tests that use this Verilog backend (and that now pass because of your fixes). This means that, when you merge your feature into main, these tests will make sure that no one else breaks your Verilog patch later on. More formally, you will have a set of tests *P*, *S⊂P*; when you merge, *S* will become *P* and then *P* will be required to pass for everyone as stated in rule 1, ensuring that your feature still works properly even with their changes.

## GitHub Actions

There is an obvious problem with everything described in the previous section: running tests takes time, it is annoying and no one wants to do it. Having to wait for 30 minutes every so often for all tests to run, while you cannot even do anything else on your computer because the CPU is loaded with integration test tasks with multiple parallel workers is a very unpleasant experience. It would be ideal if we could somehow do all of this on someone else's computer, and just get a checkmark or an X to let us know if everything is okay or not.

This is where [GitHub Actions](https://docs.github.com/en/actions) comes into play. Actions is a feature of GitHub which allows developers to automate various tasks related to their repository, such as building, testing, deploying etc. Generally, we use the term "workflow" to refer to any such task. 

### How Dynamatic uses GitHub Actions

The main Actions workflow for building and integration is described in `.github/workflows/ci.yml`. It runs every time a pull request into main is opened, reopened, updated (i.e. new commits are pushed to it) or marked ready for review (i.e. converted from draft PR to "regular" PR). It consists of two jobs: *check-format* and *integration*. As the name implies, the purpose of *check-format* is to ensure that the code style (formatting) is consistent (see [Formatting](Formatting.md)), while *integration* builds Dynamatic and runs integration tests.

This means that, in order to fulfill rule 0 of integration testing mentioned in the previous section, you do not need to waste your time and CPU resources; GitHub Actions will do it for you. When you open a pull request, the code will automatically be tested and you will be alerted if anything fails. For more details, see [Actions](Actions.md).
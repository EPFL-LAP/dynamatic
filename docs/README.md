# Table of Contents

1. [Installation](GettingStarted/InstallDynamatic.md)
2. [Tutorials](GettingStarted/Tutorials/Tutorials.md)
3. [Basic Example](GettingStarted/Tutorials/Introduction/Examples.md)
4. [User Guide](UserGuide)
   - [Command Reference](UserGuide/CommandReference.md)
   - [Writing C Code for Dynamatic](UserGuide/WritingHLSCode.md)
   - [Analyzing Output Files](UserGuide/AnalyzingOutputFiles.md)
   - [Verification](UserGuide/Verification.md)
   - [Optimizations](UserGuide/Optimizations.md)
   - [Dependencies](UserGuide/Dependencies.md)
5. [Advanced Build Instructions](UserGuide/AdvancedBuild.md)
   - [Gurobi Setup](UserGuide/AdvancedBuild.md#1-gurobi)
   - [Cloning](UserGuide/AdvancedBuild.md#2-cloning)
   - [Build Options](UserGuide/AdvancedBuild.md#3-building)
   - [Interactive Dataflow Circuit Visualizer](UserGuide/AdvancedBuild.md#4-interactive-dataflow-circuit-visualizer)
   - [XLS Integration](UserGuide/AdvancedBuild.md#5-enabling-the-xls-integration)
   - [Modelsim/Questa Installation](UserGuide/AdvancedBuild.md#6-modelsimquesta-installation)
6. [Developer Guide](DeveloperGuide/)
   - [Dynamatic HLS Flow](DeveloperGuide/DynamaticHLSFlow.md)
   - [Software Architecture](DeveloperGuide/SoftwareArchitecture.md)
   - [Writing Compiler Passes](DeveloperGuide/CreatingPasses/CreatingPasses.md)
   - [MLIR Primer](DeveloperGuide/MLIRPrimer.md)
   - [Optimization Tools](DeveloperGuide/OptimizationTools.md)
   - [Adding New Components](DeveloperGuide/AddNewComponent.md)
   - [Testing](DeveloperGuide/Testing.md)
   - [Development](DeveloperGuide/Development.md)  
   - [Contributing](DeveloperGuide/Contributing.md)
7. [Speculation](Speculation)
8. [Other Advanced Topics](Specs)
   - [Load Store Queues](LSQ)
   - [Buffering](Specs/Buffering)
   - [Backend](Specs/Backend.md)
   - [Compiler Intrinsics](Specs/CompilerIntrinsics.md)
   - [Circuit Interface](Specs/CircuitInterface.md)
   - [Floating Point Units](Specs/FloatingPointUnits.md)
   - [Formal Properties](Specs/FormalProperties.md)
   - [MLIR Optimization C Level](Specs/MLIROpInstantiationCLevel.md)
   - [Signal Manager](Specs/SignalManager.md)
   - [Timing Information](Specs/TimingInformation.md)
   - [Type System](Specs/TypeSystem.md)
   - [Extra Signals Type Verification](Specs/ExtraSignalsTypeVerification.md)
9. [Design Decision Proposals](DesignDecisionProposals/)



## Sections Overview
### 1. Installation
An detailed guide on how to install Dynamatic.

### 2. Tutorials
Beginner guides on how to use Dynamatic and perform some basic modifications.

### 3. Basic Example
An example of the Dynamatic flow that uses most of the commands a typical user will need with illustrations of expected outputs.

### 4. User Guide
All the information a first time user will need to navigate their way around Dynamatic to produce high performance data flow circuits to meet their unique needs.

### 5. Advanced Build Instructions
Instructions on how to:
- Customize the build process
- Set up the MILP solver
- Set up the visualizer
- Install the simulator used by Dynamatic

### 6. Developer Guide
Information specifically for developers wishing to perform more fine-grained customizations to Dynamatic. Details of the Dynamatic in-workings are provided here to give developers a good understanding of the internal structure to facilitate modification and contribution.

### 7. Speculation
Information on Speculation, how to use it in Dynamatic, and a PDF file with detailed information on the subject. 

### 8. Other Advanced Topics
A list of other advanced topics that could be relevant to developers and curious users wishing to better understand Dynamatic and some of its implementation details

### 9. Design Decision Proposals
Design proposals on different topics. This will be updated as new topics arise.
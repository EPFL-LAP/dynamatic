[Introduction](Intro.md)

# Getting Started

- [Install Dynamatic](GettingStarted/InstallDynamatic.md)
- [Tutorials](GettingStarted/Tutorials/Tutorials.md)
  - [Introduction](GettingStarted/Tutorials/Introduction/Introduction.md)
    - [Examples](GettingStarted/Tutorials/Introduction/Examples.md)
    - [Modifying Dynamatic](GettingStarted/Tutorials/Introduction/ModifyingDynamatic.md)
    - [Using Dynamatic](GettingStarted/Tutorials/Introduction/UsingDynamatic.md)
- [VM Setup](GettingStarted/VMSetup.md)

# User Guide

- [Advanced Build](UserGuide/AdvancedBuild.md)
- [Analyzing Output Files](UserGuide/AnalyzingOutputFiles.md)
- [Command Reference](UserGuide/CommandReference.md)
- [Dependencies](UserGuide/Dependencies.md)
- [Kernel Code Guidelines](UserGuide/KernelCodeGuideLines.md)
- [Optimizations And Directives](UserGuide/OptimizationsAndDirectives.md)
- [Sub Modules Guide](UserGuide/SubModulesGuide.md)
- [Verification](UserGuide/Verification.md)

# Developer Guide

- [Introductory Material]()
  - [Contributing](DeveloperGuide/IntroductoryMaterial/Contributing.md)
  - [Software Architecture](DeveloperGuide/IntroductoryMaterial/SoftwareArchitecture.md)
  - [Dynamatic HLS Flow](DeveloperGuide/IntroductoryMaterial/DynamaticHLSFlow.md)
  - [File Check Testing](DeveloperGuide/IntroductoryMaterial/FileCheckTesting.md)
  - [Tutorial: Creating Passes](DeveloperGuide/IntroductoryMaterial/Tutorials/CreatingPasses/CreatingPassesTutorial.md)
    - [1. Simplifying Merge Like Ops](DeveloperGuide/IntroductoryMaterial/Tutorials/CreatingPasses/1.SimplifyingMergeLikeOps.md)
    - [2. Writing A Simple Pass](DeveloperGuide/IntroductoryMaterial/Tutorials/CreatingPasses/2.WritingASimplePass.md)
    - [3. Greedy Pattern Rewriting](DeveloperGuide/IntroductoryMaterial/Tutorials/CreatingPasses/3.GreedyPatternRewriting.md)

  
- [Compiler Intrinsics]()
  - [Backend](DeveloperGuide/CompilerIntrinsics/Backend.md)
  - [Extra Signals Type Verification](DeveloperGuide/CompilerIntrinsics/ExtraSignalsTypeVerification.md)
  - [MLIR Op Instantiation C Level](DeveloperGuide/CompilerIntrinsics/MLIROpInstantiationCLevel.md)
  - [MLIR Primer](DeveloperGuide/CompilerIntrinsics/MLIRPrimer.md)
  - [Signal Manager](DeveloperGuide/CompilerIntrinsics/SignalManager.md)
  - [Timing Information](DeveloperGuide/CompilerIntrinsics/TimingInformation.md)
  - [Tutorial: Adding New MLIR Operations](DeveloperGuide/CompilerIntrinsics/Tutorials/AddNewMLIROperation.md)

- [Design Decision Proposals]()
  - [Add/Remove/Promote Extra Signals](DeveloperGuide/DesignDecisionProposals/AddRemovePromoteExtraSignals.md)
  - [Circuit Interface](DeveloperGuide/DesignDecisionProposals/CircuitInterface.md)
  - [Type System](DeveloperGuide/DesignDecisionProposals/TypeSystem.md)
  - [Wait Synchronization](DeveloperGuide/DesignDecisionProposals/WaitSynchronization.md)

- [Development Tools](DeveloperGuide/DevelopmentTools.md)

- [Documentation](DeveloperGuide/Documentation.md)

- [Dynamatic Features And Optimizations]()
  - [Buffering](DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/Buffering.md)
    - [MapBuf](DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/MapBuf/MapBuf.md)
      - [Blif Generator](DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/MapBuf/BlifGenerator.md)
      - [Blif Reader](DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/MapBuf/BlifReader.md)
      - [Technology Mapping](DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/MapBuf/TechnologyMapping.md)
      - [Subject Graph](DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/MapBuf/SubjectGraph.md)
  - [Formal Properties](DeveloperGuide/DynamaticFeaturesAndOptimizations/FormalProperties.md)
  - [LSQ](DeveloperGuide/DynamaticFeaturesAndOptimizations/LSQ/LSQ.md)
    - [Group Allocator](DeveloperGuide/DynamaticFeaturesAndOptimizations/LSQ/GroupAllocator.md)
    - [Port To Queue Dispatcher](DeveloperGuide/DynamaticFeaturesAndOptimizations/LSQ/PortToQueueDispatcher.md)
    - [Queue To Port Dispatcher](DeveloperGuide/DynamaticFeaturesAndOptimizations/LSQ/QueueToPortDispatcher.md)
  - [Speculation]()
    - [Adding Spec Tags to Spec Region](DeveloperGuide/DynamaticFeaturesAndOptimizations/Speculation/AddingSpecTagsToSpecRegion.md)
    - [Commit Unit Placement Algorithm](DeveloperGuide/DynamaticFeaturesAndOptimizations/Speculation/CommitUnitPlacementAlgorithm.md)
    - [Integration Tests](DeveloperGuide/DynamaticFeaturesAndOptimizations/Speculation/IntegrationTests.md)
    - [Save Commit Behavior](DeveloperGuide/DynamaticFeaturesAndOptimizations/Speculation/SaveCommitBehavior.md)

- [Specs]()
  - [Floating Point Units](DeveloperGuide/Specs/FloatingPointUnits.md)
  - [Timing Characterization](DeveloperGuide/Specs/TimingCharacterization.md)

- [XLS](DeveloperGuide/Xls/XlsIntegration.md)
  - [Lower Handshake To XLS Pass](DeveloperGuide/Xls/LowerHandshakeToXlsPass.md)

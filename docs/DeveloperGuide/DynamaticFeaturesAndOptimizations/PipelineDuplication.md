# Pipeline Duplication

## Overview
The Pipeline Duplication Pass duplicates specific program paths with hardcoded constants. By using pragmas, you can specify a variable that frequently holds a specific value (e.g., `y = 10`). The pass then generates an additional parallel path in the pipeline where that variable is treated as a constant.

This duplication alone does not alter control or program correctness, but for a performance boost should be combined with eager execution (eagerlyelastic). This allows the new duplicated path to run in parallel with the original logic, hiding latency by starting downstream calculation before the actual value of the variable is computed.

## Pragma Syntax and Usage
The duplication is controlled by defining a start and an end point. The syntax is as follow:
```
#pragma DYN predict variable = y values = [3] location = start marker = 0 type = int
#pragma DYN predict variable = z location = end marker = 0
```

#### Start Pragma (`location = start`)
Place this exactly above the operation where duplication should begin.
- **variable**: The variable to be replaced by a constant in the duplicated path.
- **values**: A list of speculative values (integers or floats) enclosed in `[]`. For each value in the list an additional speculative path will be created.
- **type**: The exact data type of the values in the list. Possible are `int`, `int32_t`, `int64_t`, `float`, `double`. This has to match the type of the variable that should be replaced.
- **marker**: An integer ID to group a start pragma with its corresponding end pragma(s).

#### End Pragma (`location = end`) 
Place this after the last operation you wish to duplicate.
- **variable**: The result of the final operation in the duplicated sequence.
- **marker**: To match to the corresponding start pragma's ID.


### Constraints & "Well-Defined" Paths
For the duplication to succeed, the graph (the operations that will be duplicated) between the start and end must be well-defined. The pass performs a DFS traversal from the start operation; every path encountered must eventually terminate at:
- A designated **end** pragma
- A **store** operation

If the pass encounters a branch that never reaches one of these "sinks", the transformation will not work and the compiler will exit with an error.

### Example
In this example, we create two speculative paths for two different markers. Each marker defines two predicted values (`7.0` and `7.5`). 

- Marker 0: Duplicates the logic from `y` up to `z`. This includes the addition and multiplication operations.
- Marker 1: Duplicates the logic from `y` until the store to `a[i]`. Because the store operation acts as a boundary, it is not duplicated, but the logic leading into it is.

```c++
void prediction(inout_float_t a[N], inout_float_t b[N], in_float_t c) {
  for (unsigned i = 0; i < N; ++i) {
    float y = a[i];
#pragma DYN predict variable = y values = [7.0, 7.5] location = start marker = 0 type = float
    z = (y + c) * 10.0f;
#pragma DYN predict variable = z location = end marker = 0
    x = z * 3.0f;
#pragma DYN predict variable = y values = [7.0, 7.5] location = start marker = 1 type = float
    a[i] = (y - c) * 10.0f;
  }
}
```


## Pass Implementation

The transformation works by extracting the info from the pragmas and then going through all the markers and duplicating the specified paths.


### Data Structure
The data from the pragma is put into a struct `PredictionData`, which contains all of the necessary information to perform the duplication.

```c++
struct PipelineDuplicationPass::PredictionData {
  mlir::Operation *startOp; // Duplication begins here (including this operation)
  mlir::Value predInput;    // Value that will be replaced on duplicated paths
  std::vector<mlir::Operation *> endOps;  // all end operations
  mlir::ArrayAttr values;   // MLIR array with the constants
  std::string dataType;     // String representation of the type (e.g. "int")
};
```

### Overview
The core pass driver (`runDynamaticPass`) executes the transformation in four sequential phases for each prediction marker:
1. **Block Splitting**: The basic block containing the start operation is split immediately before. This isolates the original operations and the following logic into a separate block (`exitBlock`).
2. **DFS**: A Depth-first Search goes through the data-flow chain beginning at `startOp` and terminates at either a user-defined `endOp` or store operations. This tracks all operations that must be duplicated. If the DFS does not find an `endOp` or store operation at all the "leaves" of the graph, it returns an error.
3. **Cloning**: The pass iterates over the list of constants (`values`). For each constant, it inserts a comparison check and a conditional branch to ensure that we still have correct control flow. The `true` branch then generates a new block with cloned versions of the operations identified by the DFS, substituting `predInput` with the hardcoded constant.
4. **False Path**: When all of the paths with constants have been created, the last `false` branch has all of the original operations moved into the last alternative block. All of these paths then merge back into the `exitBlock`.

A high-level overview of the `runDynamaticPass` is given here:

```c++
void PipelineDuplicationPass::runDynamaticPass() {
  
  // read input data from the pragmas
  std::vector<PredictionData> pragmaData;
  readPredictMarker(pragmaData);

  // iterate over all start operations
  for (auto data : pragmaData) {
    
    // 1. restructure the blocks
    mlir::Block *targetBlock = data.startOp->getBlock();
    mlir::Block *exitBlock = targetBlock->splitBlock(startOp);

    // 2. DFS starting from the results of startOp
    collectOpsDFS(startOp->getResult(0), data.endOps, visitedOps, outsideDrivers);

    // iterate through the function to sort
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Operation &blockOp : funcOp.getOps()) {
      if (visitedOps.count(&blockOp)) {
        opsToMove.push_back(&blockOp);
      }
    }

    // 3. iterate through values list and populate the true block
    for (mlir::Attribute attr : data.values) {

      // create the branch condition
      Value constantComp = builder.create<mlir::arithConstantOp>();
      Value branchCond = builder.create<mlir::arith::CmpOp>();

      mlir::Block *trueBlock = funcOp.addBlock(); // true path
      mlir::Block *nextElseBlock;                 // false path

      // clone the operations found during the dfs to the trueBlock
      mlir::IRMapping mapper;
      mapper.map(predictInput, constantComp);
      for (mlir::Operation *origOp : opsToMove) {
        mlir::Operation *cloned = builder.clone(*origOp, mapper);

        // remap cloned operations
        for (unsigned j = 0; j < cloned->getNumResults(); j++) {
          mapper.map(origOp->getResult(j), cloned->getResult(j));
        }
      }

      // handle the outgoing values
      llvm::SmallVector<Value> trueBranchOperands;
      for (Value origOut : originalOutputs) {
        trueBranchOperands.push_back(mapper.lookup(origOut));
      }
      builder.setInsertionPointToEnd(trueBlock);
      builder.create<mlir::cf::BranchOp>(loc, exitBlock, trueBranchOperands);

      // update
      targetBlock = nextElseBlock
    }

    // 4. False path: move all operations from the exitblock to the false block
    for (Operation *origOp : opsToMove) {
      origOp->moveBefore(targetBlock, targetBlock->end());
    }
    builder.setInsertionPointToEnd(targetBlock);
    builder.create<mlir::cf::BranchOp>(loc, exitBlock, originalOutputs);
  }

  // run the push-constants pass after duplication
  pm.addPass(dynamatic::createPushConstants());
  if (failed(pm.run(modOp)))
    return signalPassFailure();
}
```

### Helper Functions

`readPredictMarker`
```c++
LogicalResult PipelineDuplicationPass::readPredictMarker(
    mlir::ModuleOp modOp, std::vector<PredictionData> &pragmaData);
```

Scans the function IR for `dynamatic.prediction_marker` operations generated by the pragmas. It maps markers with identical numerical IDs into the `PredictionData` struct. The value list is parsed with an additional function (`parseValuesList`) to match them to the correct target data type. Once all of the data has been read out, the prediction markers are erased from the module.

`collectOpsDFS`
```c++
LogicalResult PipelineDuplicationPass::collectOpsDFS(
    mlir::Value currentVal, std::vector<mlir::Operation *> endOps,
    llvm::DenseSet<mlir::Operation *> &visitedOps,
    llvm::DenseSet<mlir::Value> &outsideDrivers)
```

Traces the downstream data flow using a Depth-first Search. The recursion records all operations that it passes through and should be duplicated in `visitedOps`. To prevent unsafe duplication, the search stops along a path if it encounters either an `endOp` or a store operation. Values escaping this subgraph are appended to `outsideDrivers` to identify which results must be added as arguments to the next block.

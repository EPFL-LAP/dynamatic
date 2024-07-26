#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <functional>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// States
//===----------------------------------------------------------------------===//
class Simulator;
/// Class that represents the channel
/// I came up to the idea that as we still have Any for data members, the
/// complicated inheritance is useless for channel representation
class ValueState {
  friend class Simulator;

public:
  ValueState(Value val) : val(val) {}
  ValueState(const ValueState &) = default;
  // Overloaded operator= helps find out if the data value was updated in an
  // effort to reduce the number of copying
  ValueState &operator=(const ValueState &other) {
    valid = other.valid;
    ready = other.ready;
    if (data.has_value() && other.anyChange) {
      data = other.data;
    }
    return *this;
  }
  // To be honest, I don't like the idea of accessibility to valid and data
  // fields, but necessity of the specific function to read/write data, but it
  // help preserve the condition of value change
  void putData(const llvm::Any &newData) {
    if (newData.has_value()) {
      data = newData;
      anyChange = true;
    }
  };
  llvm::Any getData() {
    if (data.has_value())
      return data;
    else
      return {};
  }
  bool valid = false;
  bool ready = false;

private:
  Value val;
  llvm::Any data = {};
  // data member to check if data value was changed during the recent iteration
  bool anyChange = false;
  // !!
  // Here we also can declare additional signals (maybe it's a good option to
  // have a sort of vector<Any>, I don't know)
};

//===----------------------------------------------------------------------===//
// Execution Model
//===----------------------------------------------------------------------===//
/// Same to the Github issue and your code
class ExecutionModel {
public:
  ExecutionModel(Operation *op) : op(op) {}
  // reader is the map where we read valueStates from
  // writer is the map where we write outputs of the current execution model to
  // Basically, reader = inputReader & outputReader
  // writer = inputWriter & outputWRiter
  virtual void exec(bool isReset, mlir::DenseMap<Value, ValueState *> &reader,
                    mlir::DenseMap<Value, ValueState *> &writer) = 0;

  virtual ~ExecutionModel() {}

protected:
  Operation *op;
};

template <typename Op>
class OpExecutionModel : public ExecutionModel {
public:
  OpExecutionModel(Op op) : ExecutionModel(op.getOperation()) {}

protected:
  Op getOperation() { return cast<Op>(op); }
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// All these classes represent components that are required for INTERNAL
/// COMPONENT'S STRUCTURE.
/// My initial approach was to make a sort of SupportModel class and inherit
/// other support components from it, but then I realised that all of them, in
/// fact, have different sets of ports with different type options. And the
/// problem is that insValid and insReady for these components may belong to the
/// output channel of different components.
/// However you left me freedom for inner states, didn't you? :)
class Mul4Stage {
public:
  Mul4Stage(unsigned bitwidth) : bitwidth(bitwidth) {}
  // isReset == 1 means that we use the particular architecture on the
  // simulator's reset only
  // Otherwise, it's any iteration of onClkRisingEdge...
  void exec(bool isReset, bool ce, const APInt &a, const APInt &b, APInt &p) {
    if (isReset) {
      // here for example we do nothing on the reset
    } else if (ce) {
      p = a * b;
    }
  }

private:
  // In fact, some components don't need the generics due to the fact of using
  // llvm::Any for their data fields.
  unsigned bitwidth;
};

class Join {
public:
  Join(unsigned size) : size(size) {}

  void exec(bool isRst, const std::vector<bool> &insValid, bool outsReady,
            std::vector<std::reference_wrapper<bool>> &insReady,
            bool &outsValid) {
    outsValid = 1;
    for (unsigned i = 0; i < size; ++i) {
      outsValid = outsValid && insValid[i];
    }

    for (unsigned i = 0; i < size; ++i) {
      insReady[i].get() = outsReady;
      for (unsigned j = 0; j < size; ++j) {
        if (i != j) {
          insReady[i].get() = insReady[i].get() && insValid[j];
        }
      }
    }
  }

private:
  unsigned size;
};

class DelayBuffer {
public:
  DelayBuffer(unsigned size = 32) : size(size) {}
  void exec(bool isReset, bool validIn, bool readyIn, bool &validOut) {
    if (isReset) {
      if (size == 1) {
        validOut = validIn;
      } else {
        validOut = 0;
      }
    } else if (readyIn) {
      validOut = validIn;
    }
  }

private:
  unsigned size;
};

class OEHB {
public:
  OEHB(unsigned dataWidth) : dataWidth(dataWidth) {}

  void exec(bool isReset, const Any &ins, bool insValid, bool outsReady,
            bool &insReady, Any &outs, bool &outsValid) {
    if (isReset) {
      outsValid = 0;
      insReady = 1;
      outs = 0;
    } else {
      outsValid = insValid;
      insReady = !insValid || outsReady;
      if (insValid && outsReady) {
        outs = ins;
      }
    }
  }

private:
  unsigned dataWidth;
};

/// Unfortunately, Tehb_dataless is used in mergeOp, so we cannot simply leave
/// only ReturnModel:( However, maybe it's better to leave one instance of
/// return and to provide null data (and null op maybe) instead... I suppose
/// it's not the urgent problem and can be figured out later
class TEHBDataless {
public:
  TEHBDataless() {}
  void exec(bool isReset, bool insValid, bool outsReady, bool &insReady,
            bool &outsValid) {
    if (isReset) {
      insReady = 1;
    } else {
      insReady = !insValid || outsReady;
    }
    outsValid = insValid;
  }
};
//===----------------------------------------------------------------------===//
// Cocrete Execution Models
//===----------------------------------------------------------------------===//
class SinkModel : public OpExecutionModel<handshake::SinkOp> {
public:
  using OpExecutionModel<handshake::SinkOp>::OpExecutionModel;
  SinkModel(handshake::SinkOp sinkOp)
      : OpExecutionModel<handshake::SinkOp>(sinkOp), op(sinkOp) {}

  void exec(bool isReset, mlir::DenseMap<Value, ValueState *> &reader,
            mlir::DenseMap<Value, ValueState *> &writer) override {
    auto ins = op.getOperand();
    // As the example, writer[ins]->ready is insReady (as we know it's the
    // output port, so we take it from `writer` map)
    writer[ins]->ready = 1;
  }

private:
  handshake::SinkOp op;
};

class MulIModel : public OpExecutionModel<handshake::MulIOp> {
public:
  using OpExecutionModel<handshake::MulIOp>::OpExecutionModel;
  MulIModel(handshake::MulIOp muliOp)
      : OpExecutionModel<handshake::MulIOp>(muliOp), op(muliOp),
        bitwidth(cast<handshake::ChannelType>(muliOp.getLhs().getType())
                     .getDataBitWidth()),
        muliJoin(2), muliMul4Stage(bitwidth), muliDelayBuffer(3),
        muliOEHB(bitwidth) {}

  void exec(bool isReset, mlir::DenseMap<Value, ValueState *> &reader,
            mlir::DenseMap<Value, ValueState *> &writer) override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto result = op.getResult();
    auto lhsData = llvm::any_cast<APInt>(reader[lhs]->getData());
    auto rhsData = llvm::any_cast<APInt>(reader[rhs]->getData());

    bool joinValid = 0, buffValid = 0, oehbReady = 0;
    Any oehbDataIn, oehbDataOut;
    APInt resData;

    std::vector<bool> insValid{reader[lhs]->valid, reader[rhs]->valid};
    std::vector<std::reference_wrapper<bool>> insReady{writer[lhs]->ready,
                                                       writer[rhs]->ready};
    // Several iteration to simulate the processes forking in parallel (inner
    // cycle in technical task). The constant was chosen arbitrarily
    for (int i = 0; i < 10; ++i) {
      muliJoin.exec(isReset, insValid, oehbReady, insReady, joinValid);
      muliMul4Stage.exec(isReset, oehbReady, lhsData, rhsData, resData);
      writer[result]->putData(resData);
      muliDelayBuffer.exec(isReset, joinValid, oehbReady, buffValid);
      muliOEHB.exec(isReset, oehbDataIn, buffValid, reader[result]->ready,
                    oehbReady, oehbDataOut, writer[result]->valid);
    }
  }

private:
  handshake::MulIOp op;
  unsigned bitwidth;
  Join muliJoin;
  Mul4Stage muliMul4Stage;
  DelayBuffer muliDelayBuffer;
  OEHB muliOEHB;
};

class ReturnModel : public OpExecutionModel<handshake::ReturnOp> {
public:
  using OpExecutionModel<handshake::ReturnOp>::OpExecutionModel;
  ReturnModel(handshake::ReturnOp returnOp)
      : OpExecutionModel<handshake::ReturnOp>(returnOp), op(returnOp),
        returnTEHB() {

    auto temp =
        cast<handshake::ChannelType>(returnOp.getInputs().front().getType());
    datawidth = temp.getDataBitWidth();
  }

  void exec(bool isReset, mlir::DenseMap<Value, ValueState *> &reader,
            mlir::DenseMap<Value, ValueState *> &writer) override {
    // It seems (after some testing) like the array for returnOp inputs/outputs
    // consists of one value Still a bit confused by the incompatibility of
    // ReturnOp arguments/results and actual information from vhdl architecture
    // declaration

    auto ins = op.getInputs().front();
    auto outs = op.getOutputs().front();
    bool regEnable = 0, regNotFull = 0;
    Any dataReg;

    for (int i = 0; i < 10; ++i) {
      returnTEHB.exec(isReset, reader[ins]->valid, reader[outs]->ready,
                      regNotFull, writer[outs]->valid);
      if (isReset) {
        dataReg = 0;
      } else if (regEnable) {
        dataReg = reader[ins]->getData();
      }
      if (regNotFull) {
        writer[outs]->putData(reader[ins]->getData());
      } else {
        writer[outs]->putData(dataReg);
      }
      writer[ins]->ready = regNotFull;
    }
  }

private:
  handshake::ReturnOp op;
  unsigned datawidth;
  TEHBDataless returnTEHB;
};

class EndModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndModel(handshake::EndOp endOp)
      : OpExecutionModel<handshake::EndOp>(endOp), op(endOp) {
    auto temp =
        cast<handshake::ChannelType>(endOp.getInputs().front().getType());
    bitwidth = temp.getDataBitWidth();
  }

  void exec(bool isReset, mlir::DenseMap<Value, ValueState *> &reader,
            mlir::DenseMap<Value, ValueState *> &writer) override {
    auto insArr = op.getInputs();
    auto outsArr = op.getReturnValues();
    for (size_t i = 0; i < insArr.size(); ++i) {
      writer[insArr[i]]->ready =
          reader[insArr[i]]->valid && reader[outsArr[i]]->ready;
      writer[outsArr[i]]->valid = reader[insArr[i]]->valid;
      writer[outsArr[i]]->putData(reader[insArr[i]]->getData());
    }
  }

private:
  handshake::EndOp op;
  unsigned bitwidth;
};

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

class Simulator {
public:
  Simulator(handshake::FuncOp funcOp) : funcOp(funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      associateState(arg, funcOp.getLoc());
    }

    for (Operation &op : funcOp.getOps()) {
      for (OpResult res : op.getResults()) {
        associateState(res, op.getLoc());
      }
    }
  }

  template <typename Model, typename Op, typename... Args>
  void registerModel(Op op, Args &&...modelArgs) {
    // I decided to leave the raw pointers, so to avoid memory leak, it's
    // necessary to check if we already have operation inside opModels map
    if (opModels.find(op) == opModels.end()) {
      ExecutionModel *model = new Model(op);
      opModels.insert({op, model});
    }
  }

  void reset() {
    // Collect all the states
    for (auto &[op, model] : opModels) {
      model->exec(1, valueStates, newValueStates);
    }
    // Update the states which were changed
    for (auto &[val, state] : newValueStates) {
      updateValueStates(val, state);
    }
  }
  // Here there'll be
  // exec/simulate/onClockChange/I_don't_know_how_to_call_this_func almost same
  // to reset(), but with arguments and the container for results (e.g.
  // OperandValues and ResultValues from the TT)

  // Raw pointers memory hell
  ~Simulator() {
    for (auto [_, model] : opModels)
      delete model;
    for (auto [_, state] : valueStates)
      delete state;
    for (auto [_, state] : newValueStates)
      delete state;
  }

  mlir::DenseMap<Value, ValueState *> getValueStates() { return valueStates; }

private:
  // Maybe need it some day, let it be the part of simulator class
  handshake::FuncOp funcOp;
  // Map for execution models
  mlir::DenseMap<Operation *, ExecutionModel *> opModels;
  // Map that stores the valueStates we read on the current iteration (to
  // collect new outputs)
  mlir::DenseMap<Value, ValueState *> valueStates;
  // And this map is for newly collected values (we can'not update valueStates
  // during collection, because some component can read changed values then)
  mlir::DenseMap<Value, ValueState *> newValueStates;
  // !!!
  // To conclude, valueStates are the value states on our previous iteration,
  // newValueStates - the current ones

  // Register the ValueState
  void registerState(Value val, const llvm::Any &data = {}) {
    // Same point on possible memory leakage here
    if (valueStates.find(val) == valueStates.end()) {
      ValueState *oldState = new ValueState(val);
      if (data.has_value()) {
        oldState->data = data;
      }
      valueStates.insert({val, oldState});
    }
    if (newValueStates.find(val) == newValueStates.end()) {
      ValueState *newState = new ValueState(val);
      if (data.has_value()) {
        newState->data = data;
      }
      newValueStates.insert({val, newState});
    }
  }
  // Here we determine the type of the value for data member(s)
  void associateState(Value val, Location loc) {
    llvm::TypeSwitch<Type, void>(val.getType())
        .Case<handshake::ChannelType>([&](handshake::ChannelType channel) {
          llvm::TypeSwitch<Type, void>(channel.getDataType())
              .Case<IntegerType>([&](IntegerType intType) {
                llvm::Any data =
                    APInt(intType.getWidth(), 0, intType.isSigned());
                registerState(val, data);
              })
              .Case<FloatType>([&](FloatType floatType) {
                llvm::Any data = APFloat(floatType.getFloatSemantics());
                registerState(val, data);
              })
              .Default([&](auto) {
                llvm::outs() << "Channel Error, we should probably report an "
                                "error and stop";
              });
        })
        .Case<handshake::ControlType>(
            [&](handshake::ControlType controlType) { registerState(val); })
        .Default([&](auto) {
          emitError(loc) << "Value " << val
                         << " has unsupported type, we should probably "
                            "report an error and stop";
        });
  }
  // Update the valueStates
  void updateValueStates(Value val, ValueState *state) {
    valueStates[val]->operator=(*state);
    state->anyChange = false;
  }
};

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "simulator");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, arith::ArithDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Assume a single Handshake function in the module
  handshake::FuncOp funcOp = *modOp->getOps<handshake::FuncOp>().begin();

  Simulator sim(funcOp);

  /// Associate an execution model to each function
  /// BTW, don't u think it's better to move it to simulator constructor?
  for (Operation &op : funcOp.getOps()) {
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<handshake::SinkOp>([&](handshake::SinkOp sinkOp) {
          sim.registerModel<SinkModel, handshake::SinkOp>(sinkOp);
        })
        .Case<handshake::MulIOp>([&](handshake::MulIOp muliOp) {
          sim.registerModel<MulIModel, handshake::MulIOp>(muliOp);
        })
        .Case<handshake::ReturnOp>([&](handshake::ReturnOp returnOp) {
          sim.registerModel<ReturnModel, handshake::ReturnOp>(returnOp);
        })
        .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
          sim.registerModel<EndModel, handshake::EndOp>(endOp);
        })
        .Default([&](auto) { llvm::outs() << "Error" << "\n"; });
    llvm::outs() << "\n";
  }
  // sime testing output for me 2 check
  auto states = sim.getValueStates();
  for (auto &[val, state] : states) {
    llvm::outs() << val << ":";
    llvm::outs() << state->valid << " " << state->ready << " ";
    llvm::outs() << "\n";
  }
  llvm::outs() << "-----------------\n";

  sim.reset();
  for (auto &[val, state] : states) {
    llvm::outs() << val << ":\n";
    llvm::outs() << state->valid << " " << state->ready << " ";
    if (state->getData().has_value()) {
      llvm::outs() << llvm::any_cast<APInt>(state->getData());
    }
    llvm::outs() << "\n";
  }
}
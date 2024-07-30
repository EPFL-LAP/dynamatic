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
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

namespace {

//===----------------------------------------------------------------------===//
// State management
//===----------------------------------------------------------------------===//

/// Parent class to allow components to read the state of one of their inputs.
class StateReader {
public:
  virtual ~StateReader() {};
};

/// Abstract parent class to allow components to change the state of one of
/// their outputs while not applying these changes directly on the circuit
/// state.
class StateWriter {
public:
  /// Resets any buffered changed to the underlying circuit state. Probably
  /// should be called after each round of updates to the circuit state have
  /// been applied.
  virtual void reset() = 0;

  /// Applies recorded changes to the underlying circuit state. Returns whether
  /// they were any change to the state.
  virtual bool apply() = 0;

  virtual ~StateWriter() {};
};

/// Abstract parent class to represent the evolivng state associated to an SSA
/// value in the IR.
class ValueState {
public:
  ValueState(Value val) : val(val) {}

  // The differentiation below between the reader/write of a producer OR
  // consumer is because we have SSA values that are in fact bi-directional
  // (typically, valid and data downstream signals and ready upstream signal)
  // despite the fact that SSA is intrinsically uni-directional. This means that
  // the result of an operation may in fact contain readable inputs and,
  // conversely, that the operand of an operation may in fact contain writable
  // outputs.

  // All the following pure virtual methods may return nullptr to signify that
  // the value contributes no read-only (input) signal or write-only (output)
  // signal to the adjacent component's external interface.

  virtual StateReader *readAsProducer() const = 0;
  virtual StateWriter *writeAsProducer() = 0;
  virtual StateReader *readAsConsumer() const = 0;
  virtual StateWriter *writeAsConsumer() = 0;

  virtual ~ValueState() {};

protected:
  Value val;
};

/// Just a helper templated parent class for defining a state associated to a
/// specific MLIR type. Not really useful with the current code but may be
/// expanded on to eventually become useful.
template <typename Ty>
class TypedValueState : public ValueState {
public:
  TypedValueState(TypedValue<Ty> val) : ValueState(val) {}
};

/// State associated to a handshake::ChannelType, our new formal channel type
/// that represents a bundle of valid + ready + data + optional extra signals.
class ChannelState : public TypedValueState<handshake::ChannelType> {
public:
  // If there is no internal state to initialize, this allows to directly use
  // TypedValueState's constructor
  using TypedValueState<handshake::ChannelType>::TypedValueState;

  ChannelState(TypedValue<handshake::ChannelType> channel)
      : TypedValueState<handshake::ChannelType>(channel) {

    // The data need to be allocated depending on the data type, use one of the
    // nice and formal MLIR types for representing numbers
    llvm::TypeSwitch<Type, void>(channel.getType())
        .Case<IntegerType>([&](IntegerType intType) {
          data = new APInt(intType.getWidth(), 0, intType.isSigned());
        })
        .Case<FloatType>([&](FloatType floatType) {
          data = new APFloat(floatType.getFloatSemantics());
        })
        .Default([&](auto) {
          emitError(channel.getLoc())
              << "Unsuported date type " << channel.getType()
              << ", we should probably report an error and stop";
        });
  }

protected:
  bool valid = false;
  bool ready = false;
  /// This should be APInt or APFloat depending on whether the data type is
  /// integer or floating, respectively.
  llvm::Any data = nullptr;

  // Technically there should also be state for an arbitrary number of extra
  // signals here, but let's start simple

public:
  // Below is the tedious work of creating reader/write objects for channels
  // from the channel producer's or consumer's perspective. This only has to be
  // done once (here) so it's not gonna be a burden for the user, but it isn't
  // particularily pretty.

  struct ProducerReader : public StateReader {
    const bool &ready;

    ProducerReader(const ChannelState &channelState)
        : ready(channelState.ready) {}
  };

  struct ProducerWriter : public StateWriter {
    std::optional<bool> valid;
    std::optional<llvm::Any> data;

    ProducerWriter(ChannelState &channelState) : channelState(channelState) {}

    void reset() override {
      valid = std::nullopt;
      data = std::nullopt;
    }

    bool apply() override {
      bool anyChange = false;
      if (valid) {
        anyChange |= channelState.valid != *valid;
        channelState.valid = *valid;
      }
      if (data) {
        // Here we cannot directly compare channelState.data and *data because
        // they are both llvm::Any so we have to conservatively assume there is
        // a change in value. This is not ideal and leaves room for mistakes,
        // but I cannot figure out an easy way to do this properly right now
        anyChange = true;
        channelState.data = *data;
      }
      return anyChange;
    }

  private:
    ChannelState &channelState;
  };

  struct ConsumerReader : public StateReader {
    const bool &valid;
    const llvm::Any &data;

    ConsumerReader(const ChannelState &channelState)
        : valid(channelState.valid), data(channelState.data) {}
  };

  struct ConsumerWriter : public StateWriter {
    std::optional<bool> ready;

    ConsumerWriter(ChannelState &channelState) : channelState(channelState) {}

    void reset() override { ready = std::nullopt; }

    bool apply() override {
      bool anyChange = false;
      if (ready) {
        anyChange |= channelState.ready != *ready;
        channelState.ready = *ready;
      }
      return anyChange;
    }

  private:
    ChannelState &channelState;
  };

  StateReader *readAsProducer() const override {
    return new ProducerReader(*this);
  }
  StateWriter *writeAsProducer() override { return new ProducerWriter(*this); }

  StateReader *readAsConsumer() const override {
    return new ConsumerReader(*this);
  }
  StateWriter *writeAsConsumer() override { return new ConsumerWriter(*this); }
};

//===----------------------------------------------------------------------===//
// State/Model interaction
//===----------------------------------------------------------------------===//

/// Oracle to allow an operation's execution model to query for read-only
/// access to all of its inputs ports.
class InputReader {
public:
  InputReader(Operation *op, const mlir::DenseMap<Value, ValueState *> &state) {
    for (Value oprd : op->getOperands()) {
      ValueState *oprdState = state.at(oprd);
      if (StateReader *reader = oprdState->readAsConsumer())
        readers.insert({oprd, reader});
    }
    for (Value res : op->getResults()) {
      ValueState *resState = state.at(res);
      if (StateReader *reader = resState->readAsProducer())
        readers.insert({res, reader});
    }
  }

  const StateReader *getState(Value val) const {
    auto stateIt = readers.find(val);
    if (stateIt == readers.end())
      return nullptr;
    return stateIt->second;
  }

  template <typename State>
  const State *getState(Value val) const {
    const StateReader *state = getState(val);
    assert(state && "state reader does not exist");
    return static_cast<const State *>(state);
  }

  ~InputReader() {
    for (auto [_, reader] : readers)
      delete reader;
  }

protected:
  /// Maps all values adjacent to an operation (operands or results) and which
  /// contribute at least one input signal to a reader object.
  mlir::DenseMap<Value, StateReader *> readers;
};

/// Oracle to allow an operation's execution model to query for write-only
/// access to all of its output ports (technically the model could also read
/// them but that has no visible effect on the simulator).
class OutputWriter {
public:
  OutputWriter(Operation *op,
               const mlir::DenseMap<Value, ValueState *> &state) {
    for (Value oprd : op->getOperands()) {
      ValueState *oprdState = state.at(oprd);
      if (StateWriter *writer = oprdState->writeAsConsumer())
        writers.insert({oprd, writer});
    }
    for (Value res : op->getResults()) {
      ValueState *resState = state.at(res);
      if (StateWriter *writer = resState->writeAsProducer())
        writers.insert({res, writer});
    }
  }

  StateWriter *getState(Value val) const {
    auto stateIt = writers.find(val);
    if (stateIt == writers.end())
      return nullptr;
    return stateIt->second;
  }

  template <typename State>
  State *getState(Value val) const {
    StateWriter *state = getState(val);
    assert(state && "state writer does not exist");
    return static_cast<State *>(state);
  }

  ~OutputWriter() {
    for (auto [_, writer] : writers)
      delete writer;
  }

protected:
  /// Maps all values adjacent to an operation (operands or results) and which
  /// contribute at least one output signal to a writer object.
  mlir::DenseMap<Value, StateWriter *> writers;
};

//===----------------------------------------------------------------------===//
// Execution models
//===----------------------------------------------------------------------===//

using ChannelInputRead = ChannelState::ConsumerReader;
using ChannelInputWrite = ChannelState::ConsumerWriter;
using ChannelOutputRead = ChannelState::ProducerReader;
using ChannelOutputWrite = ChannelState::ProducerWriter;

/// Abtract parent class for all execution models. It is associated to an
/// operation in the IR and internally maintains mutable state to implement the
/// associated component's behavior.
class ExecutionModel {
public:
  ExecutionModel(Operation *op, InputReader &reader, OutputWriter &writer)
      : op(op), reader(reader), writer(writer) {}

  /// Called at the beginning of the simulation to serve as a "reset" signal for
  /// the component. The component should set all of its outputs to a specific
  /// value.
  void virtual reset() = 0;

  /// Called at each rising edge of the clock cycle (at which point
  /// isClkRisingEdge is true) and at which possible state change due to the
  /// combinational propagation of signals (at which point isClkRisingEdge is
  /// false) to "execute" the component. This may change the component's
  /// internal state and/or the state of its output signals.
  void virtual exec(bool isClkRisingEdge) = 0;

  /// Seems weird that the execution model itself is in charge of deleting its
  /// reader and write which it doesn't allocate itself but the alternatives I
  /// can think about are not much better.
  virtual ~ExecutionModel() {
    delete &reader;
    delete &writer;
  }

protected:
  /// The associated MLIR operation.
  Operation *op;
  /// Permanent reference to the subset of the circuit state that the component
  /// can read.
  const InputReader &reader;
  /// Permanent reference to the subset of the circuit state that the component
  /// can write.
  const OutputWriter &writer;

  // Below are helper functions to create related reader/writer states for
  // dataflow channels, which is otherwise fantastically annoying.

  void getChannelInputRW(Value val, const ChannelInputRead *&read,
                         ChannelInputWrite *&write) {
    read = reader.getState<ChannelInputRead>(val);
    write = writer.getState<ChannelInputWrite>(val);
  }

  void getChannelInputsRW(ValueRange values,
                          SmallVector<const ChannelInputRead *> &reads,
                          SmallVector<ChannelInputWrite *> &writes) {
    for (Value val : values)
      getChannelInputRW(val, reads.emplace_back(), writes.emplace_back());
  }

  void getChannelOutputRW(Value val, const ChannelOutputRead *&read,
                          ChannelOutputWrite *&write) {
    read = reader.getState<ChannelOutputRead>(val);
    write = writer.getState<ChannelOutputWrite>(val);
  }

  void getChannelOutputsRW(ValueRange values,
                           SmallVector<const ChannelOutputRead *> &reads,
                           SmallVector<ChannelOutputWrite *> &writes) {
    for (Value val : values)
      getChannelOutputRW(val, reads.emplace_back(), writes.emplace_back());
  }
};

/// Abstract parent class for execution models specialized on a particular
/// operation type, allowing child models to directly retrieve the concrete
/// operation they are attached to (as opposed to an Operation*).
template <typename Op>
class OpExecutionModel : public ExecutionModel {
public:
  OpExecutionModel(Op op, InputReader &reader, OutputWriter &writer)
      : ExecutionModel(op.getOperation(), reader, writer) {}

protected:
  Op getOperation() { return cast<Op>(op); }
};

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

class Simulator {
public:
  Simulator(handshake::FuncOp funcOp) {
    // Associate a default state to each value in the function. Formally
    // speaking these states are invalid when we construct the simulator, they
    // only become valid after components are associated to an execution model
    // and reset() is called on each of them. I'm not sure this is the best or
    // even correct way to do this, we may need a way to represent a "null
    // state" at the beginning of the simulation.

    for (BlockArgument arg : funcOp.getArguments()) {
      // These block arguments should come from the "testbench" (the logic for
      // that is not included in the code)
      associateState(arg, funcOp.getLoc());
    }

    for (Operation &op : funcOp.getOps()) {
      for (OpResult res : op.getResults()) {
        associateState(res, op.getLoc());
      }
    }
  }

  /// Registers a model associated to a concret operatio ntype, creating input
  /// reader and output writer in the process. We allow an arbitrary number of
  /// arguments to be forwarded to the execution model's constructor to allow
  /// users to have programatic control inside a single execution model.
  template <typename Model, typename Op, typename... Args>
  void registerModel(Op op, Args &&...modelArgs) {
    InputReader *reader = new InputReader(op, valueStates);
    OutputWriter *writer = new OutputWriter(op, valueStates);
    ExecutionModel *model =
        new Model(op, *reader, *writer, std::forward<Args>(modelArgs)...);
    opModels.insert({op, model});
  }

  ~Simulator() {
    for (auto [_, model] : opModels) {
      delete model;
    }
    for (auto [_, state] : valueStates)
      delete state;
  }

private:
  /// Maps every operation to its execution model (which itself refences the
  /// operation's input reader and output writer)
  mlir::DenseMap<Operation *, ExecutionModel *> opModels;

  /// Maps every value in the function under simulation to its abstract state.
  mlir::DenseMap<Value, ValueState *> valueStates;

  /// Use during construction only.
  template <typename State, typename Ty>
  void registerState(Value val, Ty type) {
    ValueState *state = new State(cast<TypedValue<Ty>>(val));
    valueStates.insert({val, state});
  }

  /// Use during construction only.
  void associateState(Value val, Location loc) {
    llvm::TypeSwitch<Type, void>(val.getType())
        .Case<handshake::ChannelType>([&](handshake::ChannelType channelType) {
          registerState<ChannelState>(val, channelType);
        })
        .Default([&](auto) {
          emitError(loc) << "Value " << val
                         << " has unsupported type, we should probably "
                            "report an error and stop";
        });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Example model and main
//===----------------------------------------------------------------------===//

using BinaryCompFunc = std::function<llvm::Any(llvm::Any, llvm::Any)>;

template <typename Op>
class GenericBinaryOpModel : public OpExecutionModel<Op> {
public:
  GenericBinaryOpModel(Op op, InputReader &reader, OutputWriter &writer,
                       const BinaryCompFunc &callback, unsigned latency)
      : OpExecutionModel<Op>(op, reader, writer), callback(callback),
        latency(latency) {

    this->getChannelInputRW(op.getLhs(), lhsRead, lhsWrite);
    this->getChannelInputRW(op.getRhs(), rhsRead, rhsWrite);
    this->getChannelOutputRW(op.getResult(), resultRead, resultWrite);
  }

  void reset() override {
    // Whatever
  }

  void exec(bool isClkRisingEdge) override {
    // Compute the new result based on the current data input
    llvm::Any newResult = callback(lhsRead->data, rhsRead->data);

    // Use the standard join semantics to decide whether to put this result in
    // an internal register that will eventually end up (with a delay determined
    // by the latency) or directly on the output (if latency is 0)

    // All of this is dependent on the valid/ready state of both inputs and the
    // result of course
  }

private:
  const BinaryCompFunc &callback;
  unsigned latency;

  const ChannelInputRead *lhsRead, *rhsRead;
  ChannelInputWrite *lhsWrite, *rhsWrite;

  ChannelOutputWrite *resultWrite;
  const ChannelOutputRead *resultRead;
};

class MuxModel : public OpExecutionModel<handshake::MuxOp> {
public:
  // If there is no internal state to initialize, this allows to directly use
  // OpExecutionModel's constructor
  using OpExecutionModel<handshake::MuxOp>::OpExecutionModel;

  MuxModel(handshake::MuxOp muxOp, InputReader &reader, OutputWriter &writer)
      : OpExecutionModel<handshake::MuxOp>(muxOp, reader, writer),
        width(muxOp.getResult().getType().getIntOrFloatBitWidth()) {
    // Do any work here to initialize the component's internal state

    // What's below is NOT the component's internal state and as such is not
    // strictly necessary but it makes the code in exec nicer to write. I don't
    // like that people would have to do this manual stuff for every component
    // if they want to make their exec nice but this is the best I can come up
    // with.

    getChannelInputRW(muxOp.getSelectOperand(), selRead, selWrite);
    getChannelInputsRW(muxOp.getDataOperands(), dataRead, dataWrite);
    getChannelOutputRW(muxOp.getResult(), resRead, resWrite);
  }

  void reset() override {
    resWrite->valid = false;
    resWrite->data = APInt(width, 0);
  }

  void exec(bool isClkRisingEdge) override {
    // Doing random things to see whether my API is nice or not

    // For simplicity I assume that the dataflow channel data type is some kind
    // of i<X> so that I can cast it to APInt safely without checking. A mux
    // would in reality never need to interpret its data operands/result's
    // actual value so this is not a real problem anyway

    // I want to read my select input
    if (selRead->valid) {
      // Do something when my select input is valid; just print my data value
      auto selData = any_cast<APInt>(selRead->data);
      llvm::errs() << "Data is " << selData.getSExtValue() << "\n";
    }

    // selRead.ready <-- this is disallowed on purpose because ready is an
    // upstream signal and an upstream signal on the input cannot be read. It's
    // however possible to set the ready signal's value through the writer
    selWrite->ready = true;

    // I want to write my data output
    resWrite->valid = true;
    resWrite->data = APInt(width, 42);
  }

private:
  const unsigned width;

  const ChannelInputRead *selRead;
  ChannelInputWrite *selWrite;

  SmallVector<const ChannelInputRead *> dataRead;
  SmallVector<ChannelInputWrite *> dataWrite;

  ChannelOutputWrite *resWrite;
  const ChannelOutputRead *resRead;
};

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
  for (Operation &op : funcOp.getOps()) {
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<handshake::MulIOp>([&](handshake::MulIOp mulIOp) {
          BinaryCompFunc callback = [](llvm::Any lhs, llvm::Any rhs) {
            APInt mul = any_cast<APInt>(lhs) * any_cast<APInt>(rhs);
            return mul;
          };
          sim.registerModel<GenericBinaryOpModel<handshake::MulIOp>>(
              mulIOp, callback, 4);
        })
        .Case<handshake::MuxOp>(
            [&](handshake::MuxOp muxOp) { sim.registerModel<MuxModel>(muxOp); })
        .Default(
            [&](auto) {
              op.emitError()
                  << "Unsupported operation, we should probably report an "
                     "error and stop.";
            });
  }
}

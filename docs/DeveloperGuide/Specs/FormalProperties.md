# Formal Properties Infrastructure

This document describes the infrastructure for supporting formal properties in Dynamatic, focusing on the design decisions, implementation strategy, and intended usage. This infrastructure is used to express circuit-level runtime properties, primarily to enable formal verification via model checking.

## Overview

The infrastructure introduces a compiler pass called `annotate-properties`, which collects formal properties information from the Handshake IR, and serializes them to a shared .json database for other tools to consume (e.g., model checkers, code generators, etc.). This infrastructure is built to express "runtime" properties, which in the context of HLS mean properties that will appear in the circuit (or in the SMV model), and will be checked only during simulation (or model checking). This infrastructure does NOT support compile-time checks. These checks should be carried out through the MLIR infrstructure.

## Properties

Properties are defined as derived classes of `FormalProperty`. The `FormalProperty` class contains the base information common to all properties and should not be modified when introducing new kinds of properties.

The base fields are:
- `type`: Categorizes the formal property (currently: aob, veq).
- `tag`: Purpose of the property (e.g., opt for optimization, invar for invariants).
- `check`: Outcome of formal verification (true, false, or unchecked).

Any additional fields required for specific property types can—and should—be implemented in the derived classes. We intentionally allow complete freedom in defining these extra fields, as the range of possible properties is broad and they often require different types of information.

The only design principle when adding these extra fields is that they must be as complete as possible. The `annotate-properties` pass should be the only place in the code where the MLIR is analyzed to create properties. No further analysis should be needed by downstream tools to understand a property; they should only need to assemble the information already provided by the property object.

Formal properties are stored in a shared JSON database, with each property entry following this schema:

```
{
  "check": "unchecked",              // Model checker result: "true", "false", or "unchecked"
  "id": 0,                           // Unique property identifier
  "info": {                          // Property-specific information for RTL/SMV generation
    "owner": "fork0",
    "owner_channel": "outs_0",
    "owner_index": 0,
    "user": "constant0",
    "user_channel": "ctrl",
    "user_index": 0
  },
  "tag": "opt",                      // Property tag: "opt", "invar", "error", etc.
  "type": "aob"                      // Type: "aob" (absence of back-pressure), "veq" (valid equivalence), ...
}
```

## Adding a new property


The main goal of this infrastructure is to support the integration of as many formal properties as possible, so we have designed the process to be as simple and extensible as possible.

To illustrate how a new property can be integrated, we take an example from the paper [Automatic Inductive Invariant Generation for Scalable Dataflow Circuit Verification](https://dynamo.ethz.ch/wp-content/uploads/sites/22/2023/10/Xu_IWLS23_Inductive_Invariants.pdf).

> [!NOTE]
>  This is intended as a conceptual illustration of how to add new properties to the system, not a step-by-step tutorial. Many implementation details are intentionally left out. The design decisions presented here are meant for illustration purposes, not necessarily as the optimal solution for this particular problem.

In this example, we want to introduce a new invariant that states:
"for any fork the number of outptus that are sent state must be saller than the total number of fork outputs".

As is often the case with new properties, this one introduces requirements not previously encountered. Specifically, it refers to a state variable named "sent" inside an operation, which is not represented in the IR at all. We'll now explore one possible approach to handling this scenario.

> [!NOTE]
> If you decide to implement this or a different approach, please remember to update this documentation accordingly.


### Define your derived class

At this stage, you should define all the information needed for downstream tools to fully understand and process the property. It might be difficult at first to determine all the required fields, but that’s okay — you can always revise the class later by adding or removing fields as needed.

```
class MyNewInvariant : public FormalProperty {
public:
  // Basic getters
  std::string getOperation() { return operation; }
  unsigned getSize() { return size; }
  std::string getSignalName( unsigned idx ) { return signalNames[i]; }

  // Serializer and deserializer declarations
  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<MyNewInvariant> fromJSON(const llvm::json::Value &value,
                                               llvm::json::Path path);
  // Default constructor and destructor
  MyNewInvariant() = default;
  ~MyNewInvariant() = default;

  // Standard function used to recognize the type during downcasting
  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::MY_NEW_TYPE;
  }

  // New fields
private:
  std::string operation;
  unisgned size;
  std::vector<std::string> signalNames;
};
```

### Implement serialization and deserialization methods

Serialization and deserialization methods should be easy to implement once the fields for the derived class are decided. For our example they will look like this:

```
llvm::json::Value MyNewInvariant::extraInfoToJSON() const {
    llvm::json::Array namesArray;
    for (const auto &item : namesArray) {
        namesArray.push_back(item);
    } 
    
    return llvm::json::Object({{"operation", operation},
                             {"size", size},
                             {"signal_names", namesArray}});
}
```

```
std::unique_ptr<MyNewInvariant>
MyNewInvariant::fromJSON(const llvm::json::Value &value, llvm::json::Path path) {
  auto prop = std::make_unique<MyNewInvariant>();

  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  if (!mapper || !mapper.mapOptional("operation", prop->operation) ||
      !mapper.mapOptional("size", prop->size) ||
      !mapper.mapOptional("signal_names", namesArray))
    return nullptr;

  // parse namesArray to a vector of strings

  return prop;
}
```
### Implement the constructor

This is the most important method of your formal porperty class. The contructor is responsible for creating the property and extracting the information from MLIR so that it can be easily assembled by any downstream tool later. For our example the constructur will look like this:

```
MyNewInvariant::MyNewInvariant(unsigned long id, TAG tag, const Operation& op)
    : FormalProperty(id, tag, TYPE::MY_NEW_TYPE) {
  handshake::PortNamer namer1(&op);

  operation = getUniqueName(&op).str();
  size = op->getSize();
  for (int i = 0; i < size; i++)
    signalNames.push_back("sent_" + to_string(i));
}
```

### Update the `annotate-properties` pass to add your property

Define your annotation function and add it to the `runDynamaticPass` method:

```
LogicalResult
HandshakeAnnotatePropertiesPass::annotateMyNewInvariant(ModuleOp modOp){

  for ( /* every fork in the circuit */ ){
    // do something to the fork

    // create your property
    MyNewInvariant p(uid, FormalProperty::TAG::INVAR, op);
    propertyTable.push_back(p.toJSON());
          uid++;
  }

  return success();
}
```

Accessing a state in SMV that doesn't exist is obviously impossible. Therefore one approach could be to add an hardware parameter that will inform the SMV generator to define a state called `sent` so that it can be accessible outside of the operation.

For example the generated SMV code will look like this:

```
MODULE fork (ins_0, ins_0_valid, outs_0_ready, outs_1_ready)

  -- fork logic

  DEFINE sent_0 := ...;
  DEFINE sent_1 := ...;
```


### Update the backend with your new property

Now it's time to define how the property will be written to the output file. In the `export-rtl.cpp` file we need to modify the `createProperties` function to take into consideration our new properties when reading the database:

```
if (llvm::isa<MyNewInvariant>(property.get())) {
      auto *p = llvm::cast<MyNewInvariant>(property.get());

      // assemble the property
      std::string s = p->getOperation + "." + p.getSignalName(0);
      for (int i = 1: i < p->getSize(); i++){
        s += " + " + p->getOperation + "." + p.getSignalName(0);
      }
      s += " < " + to_string(p->getSize());

      data.properties[p->getId()] = {s, propertyTag};
    }
```

## FAQs
### Why use JSON?

- Allows decoupling between IR-level passes and later tools.
- Easily inspectable and extensible.
- Serves as a contract between compiler passes and formal verification tools.

### Can I add properties from an IR different than Handshake?

In theory this system supports adding properties at any time in the compilation flow because the .json file is always accessible, but we strongly advise against it. Properties must be fully specified by the end of compilation, and earlier IRs may lack the necessary information to construct them correctly.

If needed, a possible approach is to perform an early annotation pass that creates partial property entries (with some fields left blank), and then complete them later in Handshake via the `annotate-properties` pass. Still, whenever possible, we suggest implementing property generation directly within Handshake to avoid inconsistencies and simplify the flow.

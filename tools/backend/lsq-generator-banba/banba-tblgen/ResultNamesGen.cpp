#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using mlir::tblgen::Operator;

/// Writes the definition of getAsmResultNames for an operation,
/// which names each of its results in the printed IR
/// after the result's declaration:
///
///   void ::banba::shared::ga::ReadyGenOp::getAsmResultNames(
///       ::mlir::OpAsmSetValueNameFn setNameFn) {
///     setNameFn(getOperation()->getResult(0), "group_allocate_ready");
///   }
static void genResultNames(const Operator &op, raw_ostream &os) {
  // Start the definition
  os << "void " << op.getQualCppClassName() << "::getAsmResultNames(\n";
  os << "    ::mlir::OpAsmSetValueNameFn setNameFn) {\n";

  // For each result of the operation
  for (int index = 0; index < op.getNumResults(); ++index) {
    // Name the result at this index after its declaration
    os << "  setNameFn(getOperation()->getResult(" << index << "), \""
       << op.getResultName(index) << "\");\n";
  }

  // End the definition
  os << "}\n\n";
}

/// Generates the result names of every operation in the dialect
static bool genAllResultNames(const RecordKeeper &records, raw_ostream &os) {
  // For each operation defined in the .td files
  for (const Record *def : records.getAllDerivedDefinitions("Op")) {
    Operator op(*def);

    // Skip the operation if it does not declare the interface
    if (!op.getTrait("::mlir::OpAsmOpInterface::Trait"))
      continue;

    // Refuse a variadic result,
    // as the hardcoded indexes would not match
    if (op.getNumVariableLengthResults() > 0)
      PrintFatalError(def->getLoc(), "result names need non-variadic results");

    genResultNames(op, os);
  }
  // Tablegen's convention: returning false means "no error"
  return false;
}

static GenRegistration
    registration("gen-banba-result-names",
                 "Generate the result names of Banba's operations",
                 genAllResultNames);

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using mlir::tblgen::Operator;

/// Writes the definition of getOperandNames for an operation:
///
///   ::llvm::SmallVector<std::string>
///   banba::shared::ga::ReadyGenOp::getOperandNames() {
///     ::llvm::SmallVector<std::string> names;
///     names.push_back("ld_num_free_entries");
///     names.push_back("st_num_free_entries");
///     return names;
///   }
static void genOperandNames(const Operator &op, raw_ostream &os) {
  // Get the operation's class with its namespace, and remove the leading
  // `::`, which C++ would read as continuing the return type before it
  std::string qualClass = op.getQualCppClassName();
  StringRef opClass = StringRef(qualClass).ltrim(':');

  // Start the definition
  os << "::llvm::SmallVector<std::string>\n";
  os << opClass << "::getOperandNames() {\n";
  os << "  ::llvm::SmallVector<std::string> names;\n";

  // For each operand of the operation
  for (int index = 0; index < op.getNumOperands(); ++index) {
    StringRef name = op.getOperand(index).name;

    if (op.getOperand(index).isVariableLength()) {
      // A variadic operand holds a number of values only known
      // once the operation exists, so write a loop over them.
      // Each value is named after the operand and its position,
      // or only after the operand if there is exactly one
      os << "  for (unsigned k = 0, n = getODSOperandIndexAndLength(" << index
         << ").second; k < n; ++k)\n";
      os << "    names.push_back(n == 1 ? std::string(\"" << name
         << "\") : \"" << name << "\" + std::to_string(k));\n";
    } else {
      // Otherwise the operand is one value, named after the operand
      os << "  names.push_back(\"" << name << "\");\n";
    }
  }

  // End the definition
  os << "  return names;\n";
  os << "}\n\n";
}

/// Writes the definition of getResultNames for an operation:
///
///   ::llvm::SmallVector<std::string>
///   banba::shared::ga::ReadyGenOp::getResultNames() {
///     ::llvm::SmallVector<std::string> names;
///     names.push_back("group_allocate_ready");
///     return names;
///   }
static void genResultNames(const Operator &op, raw_ostream &os) {
  // Get the operation's class with its namespace, and remove the leading
  // `::`, which C++ would read as continuing the return type before it
  std::string qualClass = op.getQualCppClassName();
  StringRef opClass = StringRef(qualClass).ltrim(':');

  // Start the definition
  os << "::llvm::SmallVector<std::string>\n";
  os << opClass << "::getResultNames() {\n";
  os << "  ::llvm::SmallVector<std::string> names;\n";

  // For each result of the operation
  for (int index = 0; index < op.getNumResults(); ++index) {
    StringRef name = op.getResultName(index);

    if (op.getResult(index).isVariableLength()) {
      // A variadic result holds a number of values only known
      // once the operation exists, so write a loop over them.
      // Each value is named after the result and its position,
      // or only after the result if there is exactly one
      os << "  for (unsigned k = 0, n = getODSResultIndexAndLength(" << index
         << ").second; k < n; ++k)\n";
      os << "    names.push_back(n == 1 ? std::string(\"" << name
         << "\") : \"" << name << "\" + std::to_string(k));\n";
    } else {
      // Otherwise the result is one value, named after the result
      os << "  names.push_back(\"" << name << "\");\n";
    }
  }

  // End the definition
  os << "  return names;\n";
  os << "}\n\n";
}

/// Generates the port names of every operation in the dialect
static bool genPortNames(const RecordKeeper &records, raw_ostream &os) {
  // For each operation defined in the .td files
  for (const Record *def : records.getAllDerivedDefinitions("Op")) {
    Operator op(*def);

    // Skip the operation if it does not declare the interface
    if (!op.getTrait("::banba::PortNames::Trait"))
      continue;

    genOperandNames(op, os);
    genResultNames(op, os);
  }
  // Tablegen's convention: returning false means "no error"
  return false;
}

static GenRegistration
    registration("gen-banba-port-names",
                 "Generate the port names of Banba's operations",
                 genPortNames);

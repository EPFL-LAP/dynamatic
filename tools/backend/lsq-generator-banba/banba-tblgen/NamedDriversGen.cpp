#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using mlir::tblgen::Operator;

/// Writes a forward declaration of an operation's class:
///
///   namespace banba::shared::ga { class FreeEntryOp; }
static void genForwardDeclaration(const Operator &op, raw_ostream &os) {
  // Get the operation's namespace, and remove its leading `::`
  StringRef opNamespace = op.getCppNamespace();
  opNamespace.consume_front("::");

  StringRef opClass = op.getCppClassName();
  os << "namespace " << opNamespace << " { class " << opClass << "; }\n\n";
}

/// Writes the class that provides the definitions
/// for the HasNamedOperandDrivers trait.
/// The class is different per operation, as the operands are different
static void genOperandDrivers(const Operator &op, raw_ostream &os) {
  // Get the operation's class, with its namespace
  std::string opClass = op.getQualCppClassName();

  // Start the class: HasNamedOperandDrivers for this operation,
  // which inherits from MLIR's base class for traits
  os << "template <>\n";
  os << "class HasNamedOperandDrivers<" << opClass << ">\n";
  os << "    : public TraitBase<" << opClass << ", HasNamedOperandDrivers> {\n";
  os << "public:\n";

  // For each operand of the operation
  for (int index = 0; index < op.getNumOperands(); ++index) {
    // Convert the operand's name from snake_case to CamelCase
    std::string name =
        convertToCamelFromSnakeCase(op.getOperand(index).name, true);

    // Write the named driver,
    // where each named driver matches to a hardcoded index
    // corresponding to this operand
    os << "  void drive" << name << "(Value value) {\n";
    os << "    driveOperand(this->getOperation(), " << index << ", value);\n";
    os << "  }\n";
  }

  // End the class
  os << "};\n\n";
}

/// Writes the class that provides the definitions
/// for the HasNamedResultDrivers trait.
/// The class is different per operation, as the results are different
static void genResultDrivers(const Operator &op, raw_ostream &os) {
  // Get the operation's class, with its namespace
  std::string opClass = op.getQualCppClassName();

  // Start the class: HasNamedResultDrivers for this operation,
  // which inherits from MLIR's base class for traits
  os << "template <>\n";
  os << "class HasNamedResultDrivers<" << opClass << ">\n";
  os << "    : public TraitBase<" << opClass << ", HasNamedResultDrivers> {\n";
  os << "public:\n";

  // For each result of the operation
  for (int index = 0; index < op.getNumResults(); ++index) {
    // Convert the result's name from snake_case to CamelCase
    std::string name =
        convertToCamelFromSnakeCase(op.getResultName(index), true);

    // Write the named driver,
    // where each named driver matches to a hardcoded index
    // corresponding to this result
    os << "  void drive" << name << "(Value value) {\n";
    os << "    driveResult(this->getOperation(), " << index << ", value);\n";
    os << "  }\n";
  }

  // End the class
  os << "};\n\n";
}

/// Generates the named drivers of every operation in the dialect
static bool genNamedDrivers(const RecordKeeper &records, raw_ostream &os) {
  // For each operation defined in the .td files
  for (const Record *def : records.getAllDerivedDefinitions("Op")) {
    Operator op(*def);
    // Skip the operation if it is not declared with the trait
    if (!op.getTrait("::banba::OpTrait::HasNamedOperandDrivers"))
      continue;

    // Refuse an operation with a variadic operand or result,
    // as the hardcoded indexes would not match
    if (op.isVariadic())
      PrintFatalError(def->getLoc(),
                      "named drivers need non-variadic operands and results");

    // Forward declare the operation's class,
    // so we can reference it before it is defined
    genForwardDeclaration(op, os);

    // Write the operation's two classes of named drivers,
    // inside the namespace of the dialect's traits
    os << "namespace banba::OpTrait {\n\n";
    genOperandDrivers(op, os);
    genResultDrivers(op, os);
    os << "} // namespace banba::OpTrait\n\n";
  }
  // Tablegen's convention: returning false means "no error"
  return false;
}

static GenRegistration
    registration("gen-banba-named-drivers",
                 "Generate the named drivers of Banba's operations",
                 genNamedDrivers);

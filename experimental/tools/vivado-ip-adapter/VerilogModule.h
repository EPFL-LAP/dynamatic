
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

class Statement {
public:
  virtual ~Statement() = default;
  virtual void emit(raw_indented_ostream &os, int indent = 2) const = 0;
};

class NonBlockingAssign : public Statement {
  std::string lhs, rhs;

public:
  NonBlockingAssign(std::string lhs, std::string rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  void emit(raw_indented_ostream &os, int indent = 2) const override {
    os << std::string(indent, ' ') << lhs << " <= " << rhs << ";\n";
  }
};

class IfElseBlock : public Statement {
  std::string condition;
  std::vector<std::unique_ptr<Statement>> ifBody;
  std::vector<std::unique_ptr<Statement>> elseBody;

public:
  IfElseBlock(std::string condition) : condition(std::move(condition)) {}

  IfElseBlock &addIf(std::unique_ptr<Statement> stmt) {
    ifBody.push_back(std::move(stmt));
    return *this;
  }

  IfElseBlock &addElse(std::unique_ptr<Statement> stmt) {
    elseBody.push_back(std::move(stmt));
    return *this;
  }

  void emit(raw_indented_ostream &os, int indent = 2) const override {
    os << std::string(indent, ' ') << "if (" << condition << ")\n";
    for (const auto &stmt : ifBody)
      stmt->emit(os, indent + 2);
    if (!elseBody.empty()) {
      os << std::string(indent, ' ') << "else\n";
      for (const auto &stmt : elseBody)
        stmt->emit(os, indent + 2);
    }
  }
};

class AlwaysBlock {
  std::string sensitivity;
  std::vector<std::unique_ptr<Statement>> body;

public:
  AlwaysBlock(std::string sensitivity) : sensitivity(std::move(sensitivity)) {}

  AlwaysBlock &add(std::unique_ptr<Statement> stmt) {
    body.push_back(std::move(stmt));
    return *this;
  }

  void emit(raw_indented_ostream &os) const {
    os << "  always @(" << sensitivity << ") begin\n";
    for (const auto &stmt : body)
      stmt->emit(os, 4);
    os << "  end\n";
  }
};

class Instance {
  std::string moduleName;
  std::string instanceName;
  std::vector<std::pair<std::string, std::string>> connections; // .port(signal)

public:
  Instance(std::string module, std::string inst)
      : moduleName(std::move(module)), instanceName(std::move(inst)) {}

  Instance &connect(const std::string &port, const std::string &signal) {
    connections.emplace_back(port, signal);
    return *this;
  }

  void emit(raw_indented_ostream &os) const {
    os << moduleName << " " << instanceName << " (\n";
    os.indent();
    for (size_t i = 0; i < connections.size(); ++i) {
      const auto &[port, sig] = connections[i];
      os << "." << port << "(" << sig << ")";
      if (i != connections.size() - 1)
        os << ",";
      os << "\n";
    }
    os.unindent();
    os << ");\n";
  }
};

class VerilogModule {
public:
  VerilogModule(const std::string &name) : name(name) {}

  VerilogModule &port(const std::string &name, int width,
                      bool isOutput = false) {
    ports.emplace_back(isOutput ? "output" : "input", width, name);
    return *this;
  }

  VerilogModule &wire(const std::string &name, int width = 1) {
    wires.emplace_back(width, name);
    return *this;
  }

  VerilogModule &reg(const std::string &name, int width = 1) {
    regs.emplace_back(width, name);
    return *this;
  }

  VerilogModule &assign(const std::string &lhs, const std::string &rhs) {
    assigns.emplace_back(lhs, rhs);
    return *this;
  }

  VerilogModule &instantiate(const Instance &inst) {
    instances.push_back(inst);
    return *this;
  }

  VerilogModule &always(AlwaysBlock blk) {
    alwaysBlocks.push_back(std::move(blk));
    return *this;
  }

  void emit(raw_indented_ostream &os) const {
    os << "module " << name << "(";
    for (size_t i = 0; i < ports.size(); ++i) {
      os << ports[i].name;
      if (i < ports.size() - 1)
        os << ", ";
    }
    os << ");\n";
    os.indent();

    for (const auto &p : ports) {
      os << p.dir;
      if (p.width > 1)
        os << " [" << (p.width - 1) << ":0]";
      os << " " << p.name << ";\n";
    }

    for (const auto &w : wires) {
      os << "wire ";
      if (w.width > 1)
        os << "[" << (w.width - 1) << ":0] ";
      os << w.name << ";\n";
    }

    for (const auto &r : regs) {
      os << "reg ";
      if (r.width > 1)
        os << "[" << (r.width - 1) << ":0] ";
      os << r.name << ";\n";
    }

    for (const auto &a : assigns) {
      os << "assign " << a.lhs << " = " << a.rhs << ";\n";
    }

    for (const auto &blk : alwaysBlocks)
      blk.emit(os);

    for (const auto &inst : instances)
      inst.emit(os);

    os.unindent();
    os << "endmodule\n";
  }

private:
  struct Port {
    std::string dir;
    int width;
    std::string name;
    Port(const std::string &d, int w, const std::string &n)
        : dir(d), width(w), name(n) {}
  };

  struct Wire {
    int width;
    std::string name;
    Wire(int w, const std::string &n) : width(w), name(n) {}
  };

  struct Assign {
    std::string lhs, rhs;
    Assign(const std::string &l, const std::string &r) : lhs(l), rhs(r) {}
  };

  struct Reg {
    int width;
    std::string name;
    Reg(int w, const std::string &n) : width(w), name(n) {}
  };

  std::string name;
  std::vector<Port> ports;
  std::vector<Wire> wires;
  std::vector<Reg> regs;
  std::vector<Assign> assigns;
  std::vector<Instance> instances;
  std::vector<AlwaysBlock> alwaysBlocks;
};
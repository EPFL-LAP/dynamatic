
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

class Instance {
  std::string moduleName;
  std::string instanceName;
  std::vector<std::pair<std::string, std::string>> connections; // .port(signal)
  std::vector<unsigned> params;

public:
  Instance(std::string module, std::string inst, std::vector<unsigned> p = {})
      : moduleName(std::move(module)), instanceName(std::move(inst)),
        params(std::move(p)) {}

  Instance &connect(const std::string &port, const std::string &signal) {
    connections.emplace_back(port, signal);
    return *this;
  }

  void emit(raw_indented_ostream &os) const {
    os << moduleName << " ";

    if (!params.empty()) {
      os << "#(";
      for (size_t i = 0; i < params.size(); ++i) {
        os << params[i];
        if (i != params.size() - 1)
          os << ", ";
      }
      os << ") ";
    }

    os << instanceName << " (\n";
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

  void emit(raw_indented_ostream &os) const {
    os << "module " << name << "(";
    os.indent();
    for (size_t i = 0; i < ports.size(); ++i) {
      os << ports[i].name;
      if (i < ports.size() - 1)
        os << ",\n";
    }
    os.unindent();
    os << "\n);\n";
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

    for (const auto &inst : instances)
      inst.emit(os);

    os.unindent();
    os << "endmodule\n";
  }

  std::string getName() { return this->name; }

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
};
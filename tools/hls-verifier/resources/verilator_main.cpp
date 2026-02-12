// DESCRIPTION: Verilator testbench simulation
//
// Based on
// https://github.com/verilator/verilator/blob/master/examples/make_tracing_c/sim_main.cpp
// by Wilson Snyder, 2017.
//
//======================================================================

// For std::unique_ptr
#include <memory>

// Include common routines
#include <verilated.h>

// Include model header, generated from Verilating "tb_<kernel name>.v"
#include "Vtb.h"

// Legacy function required only so linking works on Cygwin and MSVC++
double sc_time_stamp() { return 0; }

int main(int argc, char **argv) {

  // Create logs/ directory in case we have traces to put under it
  Verilated::mkdir("logs");

  // Using unique_ptr is similar to
  // "VerilatedContext* contextp = new VerilatedContext" then deleting at end.
  const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
  // Do not instead make Vtop as a file-scope static variable, as the
  // "C++ static initialization order fiasco" may cause a crash

  // Set debug level, 0 is off, 9 is highest presently used
  // May be overridden by commandArgs argument parsing
  contextp->debug(0);

  // Peak number of threads the model will use
  // (e.g. match the --threads setting of the Verilation)
  contextp->threads(1);

  // Randomization reset policy
  // May be overridden by commandArgs argument parsing
  contextp->randReset(2);

  // Verilator must compute traced signals
  contextp->traceEverOn(true);

  // Pass arguments so Verilated code can see them, e.g. $value$plusargs
  // This needs to be called before you create any model
  contextp->commandArgs(argc, argv);

  // Construct the Verilated model, from Vtb.h generated from Verilating
  // "tb_<kernel name>.v". Using unique_ptr is similar to "Vtb* tb = new Vtb"
  // then deleting at end. "tb" will be the hierarchical name of the module.
  const std::unique_ptr<Vtb> tb{new Vtb{contextp.get(), "tb"}};

  // Simulate until $finish
  while (!contextp->gotFinish()) {

    contextp->timeInc(1); // 1 timeprecision period passes...

    // Evaluate model
    // (If you have multiple models being simulated in the same
    // timestep then instead of eval(), call eval_step() on each, then
    // eval_end_step() on each. See the manual.)
    tb->eval();
  }

  // Final model cleanup
  tb->final();

// Coverage analysis (calling write only after the test is known to pass)
#if VM_COVERAGE
  Verilated::mkdir("logs");
  contextp->coveragep()->write("logs/coverage.dat");
#endif

  // Final simulation summary
  contextp->statsPrintSummary();

  // Return good completion status
  // Don't use exit() or destructor won't get called
  return 0;
}
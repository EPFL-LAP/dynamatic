//===- MILP.h - Support for defining/solving MILPs --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common infrastructure for MILP-based algorithms (requires Gurobi). This
// mainly declares the abstract `MILP` class, which provided a unified API to
// work with MILPs.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_MILP_H
#define DYNAMATIC_SUPPORT_MILP_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "mlir/Support/LogicalResult.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
namespace dynamatic {

/// Returns a string describing the meaning of the passed Gurobi optimization
/// status code. Descriptions are taken from
// https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html
std::string getGurobiOptStatusDesc(int status);

/// Abstract class providing an API and some state-management logic to solve an
/// MILP. Implementors of this class should provide:
/// 1. A constructor that setups the entire MILP (objective and constraints) and
/// marks it ready for optimization (using `MILP::markReadyToOptimize`).
/// 2. A `MILP::extractResult` method to extract the desired result from the
/// MILP's solution.
///
/// The class manages the MILP's state throughout its lifetime (setup,
/// optimization, result extraction). After object construction and if setup was
/// successful (which one can verify by checking the return value of
/// `MILP::isReadyForOptimization`), `MILP::optimize` may be called to launch
/// the underlying MILP solver (the execution of which may take long, possibly
/// forever if no timeout was set). If optimization was successful,
/// `MILP::getResult` may then be called to extract the result from the MILP's
/// solution.
///
/// Gurobi's C++ API is used internally to manage the MILP.
template <typename Result>
class MILP {
  enum class State;

public:
  /// The Gurobi environment is used to create the internal Gurobi model. If a
  /// non-empty twine is provided, the `MILP::optimize` method will store the
  /// MILP model and its solution at `writeTo`_model.lp and
  /// `writeTo`_solution.json, respectively.
  MILP(GRBEnv &env, const llvm::Twine &writeTo = "")
      : model(GRBModel(env)), writeTo(writeTo.str()){};

  /// Optimizes the MILP. If a logger was provided at object creation, the MILP
  /// model and its solution are stored in plain text in its associated
  /// directory. If a valid pointer is provided, saves Gurobi's optimization
  /// status code in it after optimization.
  LogicalResult optimize(int *milpStatus = nullptr) {
    if (!isReadyForOptimization()) {
      llvm::errs() << "The MILP is not ready for optimization (reason: "
                   << getStateMessage() << ").\n";
      return failure();
    }

    // Optimize the model, possibly logging the MILP model and its solution
    if (!writeTo.empty()) {
      model.write(writeTo + "_model.lp");
      model.optimize();
      model.write(writeTo + "_solution.json");
    } else {
      model.optimize();
    }

    // Check whether we found an optimal solution or reached the time limit
    int stat = model.get(GRB_IntAttr_Status);
    if (milpStatus)
      *milpStatus = stat;
    if (stat != GRB_OPTIMAL && stat != GRB_TIME_LIMIT) {
      state = State::FAILED_TO_OPTIMIZE;
      llvm::errs() << "Buffer placement MILP failed with status " << stat
                   << ", reason:" << getGurobiOptStatusDesc(stat) << "\n";
      return failure();
    }
    state = State::OPTIMIZED;
    return success();
  }

  /// Extracts a result of the class template type from the MILP's solution
  /// derived during optimization. It is only possible to extract results
  /// successfully if setup and optimization were successful.
  LogicalResult getResult(Result &result) {
    if (state != State::OPTIMIZED) {
      llvm::errs()
          << "Buffer placements cannot be extracted from MILP (reason: "
          << getStateMessage() << ").";
      return failure();
    }

    extractResult(result);
    return success();
  }

  /// Marks the MILP as ready to be optimized. Should be called by the class
  /// constructor after successfully setting up the MILP's objective and
  /// constraints.
  void markReadyToOptimize() {
    assert(state == State::FAILED_TO_SETUP ||
           state == State::READY &&
               "can only mark MILP ready from constructor");
    state = State::READY;
  }

  /// Determines whether the MILP is in a valid state to be optimized. If this
  /// returns true, `MILP::optimize` can be called to solve the MILP.
  /// Conversely, if this returns false then a call to optimize will necessarily
  /// produce a failure.
  bool isReadyForOptimization() { return state == State::READY; };

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy constructor is deleted.
  MILP(const MILP &) = delete;

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy-assignment constructor is deleted.
  MILP &operator=(const MILP &) = delete;

  /// Virtual default destructor.
  virtual ~MILP() = default;

protected:
  /// Gurobi model holding the MILP's state.
  GRBModel model;

  /// Fills in the argument with the desired results extract from the MILP's
  /// solution. Called by `MILP::getResult` after checking that the underlying
  /// MILP model was optimized successfully. This cannot fail.
  virtual void extractResult(Result &result) = 0;

private:
  /// Denotes the internal state of the MILP.
  enum class State {
    /// Failed to setup the MILP, it cannot be optimized.
    FAILED_TO_SETUP,
    /// The MILP is ready to be optimized.
    READY,
    /// MILP optimization failed, result cannot be extracted.
    FAILED_TO_OPTIMIZE,
    /// MILP optimization succeeded, result can be extracted.
    OPTIMIZED
  };

  /// MILP's state, which changes during the object's lifetime.
  State state = State::FAILED_TO_SETUP;
  /// Path to a file at which to store the MILP's model and its solution after
  /// optimization. The model will be stored under `writeTo`_model.lp and the
  /// solution under `writeTo`_solution.json. Nothing will be stored if the
  /// string is empty.
  std::string writeTo;

  /// Returns a description of the MILP's current state.
  StringRef getStateMessage() {
    switch (state) {
    case State::FAILED_TO_SETUP:
      return "something went wrong during the creation of MILP constraints or "
             "objective, or the constructor forgot to mark the MILP ready for "
             "optimization using MILP::markReadyToOptimize";
    case State::READY:
      return "the MILP is ready to be optimized";
    case State::FAILED_TO_OPTIMIZE:
      return "the MILP failed to be optimized, check Gurobi's return value for "
             "more details on what went wrong";
    case State::OPTIMIZED:
      return "the MILP was successfully optimized";
    }
  }
};

/// Creates, optimizes, and extract results from an MILP in one go. Fails and
/// displays an error message to stderr if any step along the process fails.
/// Otherwise succeeds and stores the MILP's results in the first function
/// argument.
template <typename MILP, typename MILPRes, typename... Args>
LogicalResult solveMILP(MILPRes &milpResult, Args &&...args) {
  MILP milp = MILP(std::forward<Args>(args)...);
  if (failed(milp.optimize()) || failed(milp.getResult(milpResult)))
    return failure();
  return success();
}

} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_SUPPORT_MILP_H

#!/usr/bin/env python3
"""Solve handshake channel-count equations using a linear solver.

This script parses a generated `handshake_export.mlir` file, reconstructs the
SSA channel graph, and emits a linear system over symbolic channel-count
variables. The equations are the flow-conservation equations for handshake
components:

  * By default, the total number of tokens entering a component must equal the
    total number leaving it.
  * A few special components are handled explicitly, e.g. `fork`, `mux`,
    `cond_br`, `control_merge`, `join`, and `cmpi`.
  * `cmpi` introduces an extra symbolic split variable for the true/false
    branch counts.

The equations are solved with SymPy's linear solver, which is a standard
linear / parametric solving backend rather than a SAT/SMT backend.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import sympy as sp
except Exception as exc:  # pragma: no cover - handled in CLI
    print(f"Failed to import sympy: {exc}", file=sys.stderr)
    sys.exit(2)

CHANNEL_RE = re.compile(r"%[A-Za-z0-9_]+(?:#[0-9]+)?")
NAME_ATTR_RE = re.compile(r'handshake\.name = "([^"]+)"')
RESULT_CHANNEL_RE = re.compile(r"%([A-Za-z0-9_]+)(?::(\d+))?")

DEFAULT_COMPONENT_RULES = {
    "fork",
    "buffer",
    "gate",
    "load",
    "store",
    "mem_controller",
    "init",
    "source",
    "constant",
    "extui",
    "extsi",
    "trunci",
    "shli",
    "addi",
    "muli",
    "cmpi",
    "negi",
    "ori",
    "andi",
    "mux",
    "cond_br",
    "control_merge",
    "join",
    "dead",
}
# These operations create a Boolean value. Their output can carry a true-count,
# but traversal must not propagate that count into their operands.
BOOLEAN_BOUNDARY_OPS = {"cmpi", "andi", "ori", "xori", "not", "noti"}


class HandshakeCountSolver:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path
        self.ops: List[Dict[str, object]] = []
        self.channels: Set[str] = set()
        self.start_signal_channel: Optional[str] = None
        self.fixed_one_channels: Set[str] = set()
        self.count_vars: Dict[str, sp.Symbol] = {}
        self.true_vars: Dict[str, sp.Symbol] = {}
        self.cmp_true_vars: Dict[str, sp.Symbol] = {}
        self.cmp_false_vars: Dict[str, sp.Symbol] = {}
        self.param_symbols: Dict[str, sp.Symbol] = {}
        self.equations: List[sp.Expr] = []
        self.raw_equations: List[str] = []

    def run(self) -> int:
        self.parse()
        # print self.ops
        for op in self.ops:
            print(f"  name = {op['name']}, type = {op['type']} inputs = ({', '.join(op['inputs'])}) -> results = {', '.join(op['results'])}")
        
        self.build_channel_model()
        # print count var
        # for channel, var in self.count_vars.items():
        #     print(f"  count_var[{channel}] = {var}")
        # for channel, var in self.true_vars.items():
        #     print(f"  true_var[{channel}] = {var}")

        
        self.emit_count_constraints()

        if not self.equations:
            print("No equations were emitted")
            return 0

        solution = self.solve_linear_system()

        # write all equations in a file
        with open("equations.txt", "w") as f:
            f.write("\n".join(self.raw_equations))

        # print("Equations:")
        # for eq in self.raw_equations:
        #     print(f"  {eq}")

        if solution is None:
            print("UNSAT")
            return 1

        print("FEASIBLE")
        print("Parameterized solution:")
        print(solution)
        return 0

    def parse(self) -> None:
        with self.input_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("module") or line.startswith("}"):
                    continue
                if line.startswith("handshake.func"):
                    start_candidates = CHANNEL_RE.findall(line)
                    if start_candidates:
                        self.start_signal_channel = start_candidates[-1]
                    continue
                if "handshake.func" in line:
                    continue

                if "=" not in line and not line.startswith("end"):
                    continue

        
                

                lhs, rhs = line.split("=", 1)
                if line.startswith("end"):
                    lhs = ""
                    rhs = line
                lhs = lhs.strip()
                rhs = rhs.strip()
                comp_type = self.extract_comp_type(rhs)
                name_match = NAME_ATTR_RE.search(line)
                name = name_match.group(1) if name_match else comp_type
                input_channels = [channel for channel in CHANNEL_RE.findall(rhs)]
                result_channels = self.parse_result_channels(lhs)
                self.ops.append(
                    {
                        "type": comp_type,
                        "name": name,
                        "lhs": lhs,
                        "rhs": rhs,
                        "results": result_channels,
                        "inputs": input_channels,
                    }
                )

    def extract_comp_type(self, rhs: str) -> str:
        rhs = rhs.strip()
        match = re.match(r"^([A-Za-z0-9_]+)", rhs)
        if match:
            return match.group(1)
        match = re.search(r'"([^"]+)"', rhs)
        if match:
            return match.group(1)
        name = rhs.split()[0]
        name = name.rstrip("()")
        return name

    def parse_result_channels(self, lhs: str) -> List[str]:
        result_channels: List[str] = []
        for part in lhs.split(","):
            part = part.strip()
            if not part:
                continue
            matches = RESULT_CHANNEL_RE.findall(part)
            if not matches:
                continue
            name = matches[0][0]
            output_dimension = int(matches[0][1]) if matches[0][1] else 1
            if output_dimension == 1:
                result_channels.append(f"%{name}")
            else:
                for i in range(output_dimension):
                    result_channels.append(f"%{name}#{i}")
        return result_channels

    def build_channel_model(self) -> None:
        for op in self.ops:
            self.channels.update(op["inputs"])
            self.channels.update(op["results"])

        if self.start_signal_channel is not None:
            self.fixed_one_channels.add(self.start_signal_channel)

        for channel in sorted(self.channels):
            self.count_vars[channel] = sp.Symbol(f"[{channel}]_C")
        for channel in sorted(self.find_condition_path_channels()):
            self.true_vars[channel] = sp.Symbol(f"[{channel}]_T")

        for channel in self.fixed_one_channels:
            if channel in self.count_vars:
                self.add_linear_eqn(
                    self.count_vars[channel] - sp.Integer(1),
                    f"fixed_one:{channel}:count",
                )

    def find_condition_path_channels(self) -> Set[str]:
        """Find channels for which a true-count variable is meaningful.

        Start at mux and conditional-branch condition inputs and walk backward
        through their producers. Include a Boolean producer's output, but do
        not continue into its inputs: comparator/Boolean operands are normal
        data and do not have true-counts.
        """
        producers: Dict[str, Dict[str, object]] = {}
        worklist: List[str] = []
        for op in self.ops:
            for result in op["results"]:
                producers[result] = op
            if op["type"] in {"mux", "cond_br"} and op["inputs"]:
                worklist.append(op["inputs"][0])

        condition_channels: Set[str] = set()
        while worklist:
            channel = worklist.pop()
            if channel in condition_channels:
                continue
            condition_channels.add(channel)

            producer = producers.get(channel)
            if producer is None or producer["type"] in BOOLEAN_BOUNDARY_OPS:
                continue
            worklist.extend(producer["inputs"])

        return condition_channels
    def emit_count_constraints(self) -> None:
        for op in self.ops:
            op_type = str(op["type"])
            op_name = str(op["name"])
            inputs = list(op["inputs"])
            results = list(op["results"])

            if op_type in {"source", "constant"}:
                continue

            if op_type in {"mem_controller", "lsq"}:
                self.emit_mem_controller_lsq_constraints(op_name, inputs, results)
                continue

            if op_type == "load":
                self.emit_load_constraints(op_name, inputs, results)
                continue

            if op_type == "store":
                self.emit_store_constraints(op_name, inputs, results)
                continue

            if op_type == "mux":
                self.emit_mux_constraints(op_name, inputs, results)
                continue

            if op_type == "cond_br":
                self.emit_branch_constraints(op_name, inputs, results)
                continue

            if op_type == "control_merge":
                self.emit_control_merge_constraints(op_name, inputs, results)
                continue

            if op_type in {"fork", "buffer", "join", "end", "extui", "extsi", "trunci", "shli", "shrsi", "addi", "subi", "muli", "negi", "ori", "andi"}:
                self.emit_identity_constraints(op_name, inputs, results)
                continue

            if op_type == "cmpi":
                self.emit_comparator_constraints(op_name, inputs, results)
                continue

            else:
                print(f"Warning: Unhandled component type '{op_type}' in op '{op_name}'", file=sys.stderr)
                continue

            # if not input_counts and not output_counts:
            #     continue

            # self._add_balance_eqn(
            #     op_type,
            #     input_counts,
            #     output_counts,
            #     lhs_label=f"{op_name}:{op_type}",
            # )
            # if input_true_counts or output_true_counts:
            #     self._add_balance_eqn(
            #         f"{op_type}.true",
            #         input_true_counts,
            #         output_true_counts,
            #         lhs_label=f"{op_name}:{op_type}:true",
            #     )

    def emit_mux_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if len(inputs) < 3 or len(results) < 1:
            return
        cond = inputs[0]
        in_true = inputs[1]
        in_false = inputs[2]
        out = results[0]
        if cond in self.count_vars and in_true in self.count_vars and in_false in self.count_vars and out in self.count_vars:
            self.add_linear_eqn(self.count_vars[in_true] - self.true_vars[cond], f"{op_name}:mux:cond_vs_in_true")
            self.add_linear_eqn(self.count_vars[in_false] - self.count_vars[cond] + self.true_vars[cond], f"{op_name}:mux:cond_vs_in_false")
            self.add_linear_eqn(self.count_vars[out] - self.count_vars[cond], f"{op_name}:mux:cond_eq_out")
        if cond in self.true_vars and in_true in self.true_vars and in_false in self.true_vars and out in self.true_vars:
            self.add_linear_eqn(self.true_vars[out] - self.true_vars[in_true] - self.true_vars[in_false], f"{op_name}:mux:true")

    def emit_branch_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if len(inputs) < 2 or len(results) < 2:
            return
        cond = inputs[0]
        input = inputs[1]
        out_true = results[0]
        out_false = results[1]
        if cond in self.count_vars and input in self.count_vars and out_true in self.count_vars and out_false in self.count_vars:
            self.add_linear_eqn(self.count_vars[input] - self.count_vars[cond], f"{op_name}:branch:cond_eq_input")
            self.add_linear_eqn(self.count_vars[out_true] - self.true_vars[cond], f"{op_name}:branch:true_out")
            self.add_linear_eqn(self.count_vars[out_false] - self.count_vars[cond] + self.true_vars[cond], f"{op_name}:branch:false_out")
        if input in self.true_vars and out_true in self.true_vars and out_false in self.true_vars:
            self.add_linear_eqn(self.true_vars[input] - self.true_vars[out_true] - self.true_vars[out_false], f"{op_name}:branch:true")


    def emit_control_merge_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if len(inputs) < 3 or len(results) < 1:
            return

        in_true = inputs[0]
        in_false = inputs[1]
        select = inputs[2]
        out = results[0]

        if in_true in self.count_vars and in_false in self.count_vars and select in self.count_vars and out in self.count_vars:
            self.add_linear_eqn(self.count_vars[out]- self.count_vars[in_true]- self.count_vars[in_false],f"{op_name}:control_merge:out")
            self.add_linear_eqn(self.count_vars[select]- self.count_vars[in_true]- self.count_vars[in_false], f"{op_name}:control_merge:select_count")
        if (select in self.true_vars and in_true in self.count_vars):
            self.add_linear_eqn(self.true_vars[select] - self.count_vars[in_true], f"{op_name}:control_merge:select_true")
        if (in_true in self.true_vars and in_false in self.true_vars and out in self.true_vars):
            self.add_linear_eqn(self.true_vars[out]- self.true_vars[in_true]- self.true_vars[in_false], f"{op_name}:control_merge:true")


    # def emit_control_merge_constraints(
    #     self,
    #     op_name: str,
    #     inputs: Sequence[str],
    #     results: Sequence[str],
    #     input_counts: Sequence[sp.Symbol],
    #     output_counts: Sequence[sp.Symbol],
    #     input_true_counts: Sequence[sp.Symbol],
    #     output_true_counts: Sequence[sp.Symbol],
    # ) -> None:
    #     if len(inputs) < 2 or len(results) < 1:
    #         return
    #     out = results[0]
    #     select = inputs[-1]
    #     in_true = inputs[0]
    #     in_false = inputs[1] if len(inputs) > 1 else None
    #     if out in self.count_vars:
    #         if input_counts:
    #             self.add_linear_eqn(self.count_vars[out] - sp.Add(*input_counts), f"{op_name}:control_merge:count")
    #         if select in self.count_vars:
    #             self.add_linear_eqn(self.count_vars[select] - sp.Add(*input_counts), f"{op_name}:control_merge:select_count")
    #         if select in self.true_vars and in_true in self.count_vars:
    #             self.add_linear_eqn(self.true_vars[select] - self.count_vars[in_true], f"{op_name}:control_merge:select_true")
    #     if out in self.true_vars:
    #         if in_false is not None and in_false in self.true_vars and in_true in self.true_vars:
    #             self.add_linear_eqn(self.true_vars[out] - self.true_vars[in_true] - self.true_vars[in_false], f"{op_name}:control_merge:true")
    #         elif input_true_counts:
    #             self.add_linear_eqn(self.true_vars[out] - sp.Add(*input_true_counts), f"{op_name}:control_merge:true")

    def emit_source_constant_constraints(
        self,
        op_name: str,
        results: Sequence[str],
    ) -> None:
        for channel in results:
            if channel in self.count_vars:
                self.add_linear_eqn(self.count_vars[channel] - sp.Integer(1), f"{op_name}:source_constant:fixed_count")

    def emit_mem_controller_lsq_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if not inputs or not results:
            return
        last_input = inputs[-1]
        last_output = results[-1]
        if last_input in self.count_vars and last_output in self.count_vars:
            self.add_linear_eqn(self.count_vars[last_input] - self.count_vars[last_output], f"{op_name}:mem_controller_lsq:ctrl_end")

    def emit_store_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if len(inputs) < 2 or len(results) < 1:
            return
        addr_in = inputs[0]
        data_in = inputs[1]
        if addr_in in self.count_vars and data_in in self.count_vars:
            self.add_linear_eqn(self.count_vars[data_in] - self.count_vars[addr_in], f"{op_name}:store:address_eq_data")


    def emit_load_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if len(inputs) < 1 or len(results) < 2:
            return
        addr_in = inputs[0]
        data_out = results[1]
        if addr_in in self.count_vars and data_out in self.count_vars:
            self.add_linear_eqn(self.count_vars[data_out] - self.count_vars[addr_in], f"{op_name}:load:address_eq_data")

    def emit_identity_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        values = list(inputs) + list(results)

        if len(values) < 2:
            return

        for lhs, rhs in zip(values, values[1:]):
            if lhs in self.count_vars and rhs in self.count_vars:
                self.add_linear_eqn(self.count_vars[lhs] - self.count_vars[rhs], f"{op_name}:identity_count:{lhs}_{rhs}")

            if lhs in self.true_vars and rhs in self.true_vars:
                self.add_linear_eqn(self.true_vars[lhs] - self.true_vars[rhs], f"{op_name}:identity_true:{lhs}_{rhs}")

    # def _emit_fork_constraints(
    #     self,
    #     op_name: str,
    #     input_counts: Sequence[sp.Symbol],
    #     output_counts: Sequence[sp.Symbol],
    #     input_true_counts: Sequence[sp.Symbol],
    #     output_true_counts: Sequence[sp.Symbol],
    # ) -> None:
    #     if input_counts and output_counts:
    #         input_count = input_counts[0]
    #         for out in output_counts:
    #             self.add_linear_eqn(out - input_count, f"{op_name}:fork:count")
    #     if input_true_counts and output_true_counts:
    #         input_true = input_true_counts[0]
    #         for out in output_true_counts:
    #             self.add_linear_eqn(out - input_true, f"{op_name}:fork:true")

    # def _emit_join_constraints(
    #     self,
    #     op_name: str,
    #     input_counts: Sequence[sp.Symbol],
    #     output_counts: Sequence[sp.Symbol],
    #     input_true_counts: Sequence[sp.Symbol],
    #     output_true_counts: Sequence[sp.Symbol],
    # ) -> None:
    #     if len(input_counts) >= 2 and output_counts:
    #         self.add_linear_eqn(input_counts[0] - input_counts[1], f"{op_name}:join:count_sync")
    #         self.add_linear_eqn(input_counts[0] - output_counts[0], f"{op_name}:join:count_out")
    #     if len(input_true_counts) >= 2 and output_true_counts:
    #         self.add_linear_eqn(input_true_counts[0] - input_true_counts[1], f"{op_name}:join:true_sync")
    #         self.add_linear_eqn(input_true_counts[0] - output_true_counts[0], f"{op_name}:join:true_out")

    def emit_comparator_constraints(
        self,
        op_name: str,
        inputs: Sequence[str],
        results: Sequence[str],
    ) -> None:
        if len(results) < 1:
            return

        out = results[0]

        if out in self.count_vars:
            for value in inputs:
                if value in self.count_vars:
                    self.add_linear_eqn(self.count_vars[value] - self.count_vars[out], f"{op_name}:cmpi:count_sync")

            if out in self.true_vars:
                p_symbol = sp.Symbol(f"p_{op_name}")
                self.param_symbols[op_name] = p_symbol
                self.add_linear_eqn(self.true_vars[out] - p_symbol * self.count_vars[out], f"{op_name}:cmpi:true_probability")

    # def _add_balance_eqn(
    #     self,
    #     component_kind: str,
    #     input_counts: Sequence[sp.Symbol],
    #     output_counts: Sequence[sp.Symbol],
    #     lhs_label: str,
    # ) -> None:
    #     lhs_expr = sp.Add(*input_counts) if input_counts else sp.Integer(0)
    #     rhs_expr = sp.Add(*output_counts) if output_counts else sp.Integer(0)
    #     self.add_linear_eqn(lhs_expr - rhs_expr, f"{lhs_label}:{component_kind}")

    def add_linear_eqn(self, expr: sp.Expr, text: str) -> None:
        normalized = sp.expand(expr)
        self.equations.append(normalized)
        self.raw_equations.append(f"{text}: {sp.sstr(normalized)} = 0")

    # def solve_linear_system(self) -> Optional[str]:
    #     all_vars = list(self.count_vars.values()) + list(self.true_vars.values())
    #     all_vars += list(self.cmp_true_vars.values()) + list(self.cmp_false_vars.values())
    #     if not all_vars:
    #         return "No variables were created"

    #     unique_vars = list(dict.fromkeys(all_vars))

    #     try:
    #         solution = sp.linsolve(self.equations, unique_vars)
    #     except Exception as exc:  # pragma: no cover
    #         print(f"Linear solve failed: {exc}", file=sys.stderr)
    #         return None

    #     if solution == sp.EmptySet:
    #         return None

    #     finite = list(solution)
    #     if not finite:
    #         return None

    #     result = finite[0]
    #     if len(result) != len(unique_vars):
    #         return str(result)

    #     rendered = []
    #     for var, value in zip(unique_vars, result):
    #         rendered.append(f"{var} = {sp.sstr(value)}")
    #     return "\n".join(rendered)

    def get_largest_consistent_system(
        self,
        equations: List[sp.Eq], 
        variables: List[sp.Symbol]
    ) -> Tuple[List[sp.Eq], Optional[sp.FiniteSet]]:
        """
        Greedily keeps equations that are consistent with previously accepted ones.
        Returns (accepted_equations, final_solution).
        """
        accepted_eqs = []
        current_sol = None

        for eq in equations:
            candidate_eqs = accepted_eqs + [eq]
            try:
                sol = sp.linsolve(candidate_eqs, variables)
                # If the system remains consistent, accept the equation
                if sol != sp.EmptySet:
                    accepted_eqs = candidate_eqs
                    current_sol = sol
                else:
                    print(f"Dropped inconsistent equation: {eq}")
            except Exception as exc:
                print(f"Error evaluating equation {eq}: {exc}")

        return accepted_eqs, current_sol


    def solve_linear_system(self) -> Optional[str]:
        all_vars = list(self.count_vars.values()) + list(self.true_vars.values())
        all_vars += list(self.cmp_true_vars.values()) + list(self.cmp_false_vars.values())
        if not all_vars:
            return "No variables were created"

        unique_vars = list(dict.fromkeys(all_vars))

        # Get the largest subset of mutually consistent equations
        accepted_eqs, solution = self.get_largest_consistent_system(self.equations, unique_vars)

        if not solution or solution == sp.EmptySet:
            return None

        finite = list(solution)
        if not finite:
            return None

        result = finite[0]
        
        # Store accepted equations back if needed for debugging
        self.valid_equations = accepted_eqs 

        rendered = []
        for var, value in zip(unique_vars, result):
            rendered.append(f"{var} = {sp.sstr(value)}")
        return "\n".join(rendered)


def collect_files(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    return sorted(target.rglob("handshake_export.mlir"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a handshake export admits a feasible linear count assignment."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="Path to a handshake_export.mlir file or a directory tree to search.",
    )
    parser.add_argument(
        "--emit-equations",
        dest="emit_equations",
        default=None,
        help="Optional file path for writing the generated equations.",
    )
    args = parser.parse_args(argv)

    target = Path(args.input).resolve()
    files = collect_files(target)
    if not files:
        print("No handshake_export.mlir files were found", file=sys.stderr)
        return 2

    overall_status = 0
    for path in files:
        print(f"Checking {path}")
        try:
            solver = HandshakeCountSolver(path)
            status = solver.run()
            if args.emit_equations:
                out_path = Path(args.emit_equations)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"# {path}\n")
                    handle.write("\n".join(solver.raw_equations))
                    handle.write("\n\n")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to analyze {path}: {exc}", file=sys.stderr)
            overall_status = 1
            continue
        if status != 0:
            overall_status = 1
    return overall_status


if __name__ == "__main__":
    sys.exit(main())

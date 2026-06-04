# Extract per-(in_port, out_port) timing data and save as a JSON matrix.
import os
import re
import json

PATTERN_DELAY_INFO = "Data Path Delay:"


def extract_delay(line):
    match = re.search(r'Data Path Delay:\s+([\d.]+)ns', line)
    assert match, f"Could not find data path delay in line: {line}"
    return float(match.group(1))


def extract_single_rpt(rpt_file):
    """
    Returns the max Data Path Delay in ns. If the file is present but contains
    no Data Path Delay line (Vivado found no timing path between this in/out
    pair), returns 0.0. If the file is missing (synth failed), returns None.
    """
    if not os.path.exists(rpt_file):
        return None
    max_delay = 0.0
    with open(rpt_file, 'r') as f:
        for line in f:
            if PATTERN_DELAY_INFO in line:
                max_delay = max(max_delay, extract_delay(line))
    return max_delay


def render_matrix(unit_name, delays, bw):
    """Render the (in_port x out_port) matrix at one bitwidth as a string."""
    in_ports = list(delays.keys())
    out_ports = []
    seen = set()
    for ip in in_ports:
        for op in delays[ip]:
            if op not in seen:
                seen.add(op)
                out_ports.append(op)
    col_w = max((len(p) for p in out_ports), default=4) + 1
    row_w = max((len(p) for p in in_ports), default=4)
    lines = [f"=== {unit_name} delay matrix (bw={bw}) [ns] ==="]
    lines.append(" " * row_w + "  " + "".join(p.rjust(col_w)
                 for p in out_ports))
    for ip in in_ports:
        cells = []
        for op in out_ports:
            v = delays[ip].get(op, {}).get(bw)
            cells.append("---" if v is None else f"{v:.3f}")
        lines.append(ip.rjust(row_w) + "  " + "".join(c.rjust(col_w)
                     for c in cells))
    return "\n".join(lines)


def split_physical_port(phys):
    """
    Split a physical VHDL port name into (logical_port, signal).
    Examples:
        "dataIn"             -> ("dataIn",    "data")
        "dataIn_valid"       -> ("dataIn",    "valid")
        "dataOut_ready"      -> ("dataOut",   "ready")
        "issueCtrl_valid"    -> ("issueCtrl", "valid")
        "trigger_valid"      -> ("trigger",   "valid")
    """
    for suffix, sig in (("_valid", "valid"), ("_ready", "ready"), ("_spec", "spec")):
        if phys.endswith(suffix):
            return phys[: -len(suffix)], sig
    return phys, "data"


def extract_rpt_data(map_unit_to_list_unit_chars, json_output):
    """
    Produce two artifacts:
      1. JSON (json_output) in the edges-list shape:
         output[unit_name]["delays"] = [
           { "from": {"port", "signal"},
             "to":   {"port", "signal"},
             "samples": [ {"params": {...}, "delay": ns}, ... ] },
           ...
         ]
         spec signals and all-zero edges are dropped.
      2. A .matrix.txt next to it: per-bitwidth physical-port matrices for
         human inspection.
    """
    # Per-unit nested matrix for the human-readable view (keyed by physical port
    # names, indexed by bitwidth string).
    matrix_by_unit = {}
    # Per-unit edge map: (from_port, from_sig, to_port, to_sig) -> list of samples.
    edges_by_unit = {}

    for unit_name, list_unit_chars in map_unit_to_list_unit_chars.items():
        matrix = {}
        edges = {}
        for unit_char in list_unit_chars:
            sample_params = dict(unit_char.params)
            bw = str(sample_params.get("BITWIDTH", ""))
            pair_to_rpt = getattr(unit_char, "map_pair_to_rpt", {})
            if not pair_to_rpt:
                print(
                    "\033[91m" + f"[ERROR] No pair_to_rpt for unit {unit_name} at params={sample_params}." + "\033[0m")
                continue
            for (iport, oport), rpt_path in pair_to_rpt.items():
                delay = extract_single_rpt(rpt_path)
                if delay is None:
                    continue
                # Matrix view: keep zeros, key by bitwidth for display.
                matrix.setdefault(iport, {}).setdefault(oport, {})[bw] = delay
                # Edges view: drop spec endpoints; drop zero samples; pre-split.
                from_port, from_sig = split_physical_port(iport)
                to_port, to_sig = split_physical_port(oport)
                if from_sig == "spec" or to_sig == "spec":
                    continue
                if delay == 0.0:
                    continue
                edge_key = (from_port, from_sig, to_port, to_sig)
                edges.setdefault(edge_key, []).append(
                    {"params": sample_params, "delay": delay})

        matrix_by_unit[unit_name] = matrix
        edges_by_unit[unit_name] = edges

    # Assemble the edges-list output.
    output_data = {}
    for unit_name, edges in edges_by_unit.items():
        edge_list = []
        for (fp, fs, tp, ts), samples in edges.items():
            edge_list.append({
                "from": {"port": fp, "signal": fs},
                "to": {"port": tp, "signal": ts},
                "samples": samples,
            })
        output_data[unit_name] = {"delays": edge_list}

    # Render matrices per unit per bitwidth.
    matrix_path = json_output + ".matrix.txt"
    chunks = []
    for unit_name, matrix in matrix_by_unit.items():
        bws = sorted(
            {bw for ip in matrix for op in matrix[ip] for bw in matrix[ip][op]}, key=int)
        for bw in bws:
            block = render_matrix(unit_name, matrix, bw)
            print("\n" + block)
            chunks.append(block)
    with open(matrix_path, 'w') as f:
        f.write("\n\n".join(chunks) + "\n")

    with open(json_output, 'w') as f:
        json.dump(output_data, f, indent=2)

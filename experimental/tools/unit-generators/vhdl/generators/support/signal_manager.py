# See docs/Specs/SignalManager.md

from collections.abc import Callable


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]) -> str:
    debugging_info = f"-- Signal manager generation info: {name}, {params}\n"

    in_ports = params["in_ports"]
    out_ports = params["out_ports"]
    type = params["type"]

    if type == "normal":
        extra_signals = params["extra_signals"]
        signal_manager = _generate_normal_signal_manager(
            name, in_ports, out_ports, extra_signals, generate_inner)
    elif type == "buffered":
        extra_signals = params["extra_signals"]
        latency = params["latency"]
        signal_manager = _generate_buffered_signal_manager(
            name, in_ports, out_ports, extra_signals, generate_inner, latency)
    elif type == "concat":
        extra_signals = params["extra_signals"]
        ignore_ports = params.get("ignore_ports", [])
        signal_manager = _generate_concat_signal_manager(
            name, in_ports, out_ports, extra_signals, ignore_ports, generate_inner)
    elif type == "bbmerge":
        extra_signals = params["extra_signals"]
        index_name = params["index_name"]
        index_dir = params["index_dir"]
        signal_manager = _generate_bbmerge_signal_manager(
            name, in_ports, out_ports, index_name, extra_signals, index_dir, generate_inner)
    else:
        raise ValueError(f"Unsupported signal manager type: {type}")

    return signal_manager + debugging_info


def generate_entity(entity_name, in_ports, out_ports) -> str:
    """
    Generate entity for signal manager, based on input and output ports
    """

    # Unify input and output ports, and add direction
    unified_ports = []
    for port in in_ports:
        unified_ports.append({
            **port,
            "direction": "in"
        })
    for port in out_ports:
        unified_ports.append({
            **port,
            "direction": "out"
        })

    port_decls = []
    # Add port declarations for each port
    for port in unified_ports:
        dir = port["direction"]
        ready_dir = "out" if dir == "in" else "in"

        name = port["name"]
        bitwidth = port["bitwidth"]
        extra_signals = port.get("extra_signals", {})
        port_2d = port.get("2d", False)

        if not port_2d:
            # Usual case

            # Generate data signal if present
            if bitwidth > 0:
                port_decls.append(
                    f"    {name} : {dir} std_logic_vector({bitwidth} - 1 downto 0)")

            port_decls.append(f"    {name}_valid : {dir} std_logic")
            port_decls.append(f"    {name}_ready : {ready_dir} std_logic")

            # Generate extra signals for this input port
            for signal_name, signal_bitwidth in extra_signals.items():
                port_decls.append(
                    f"    {name}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")
        else:
            # Port is 2d
            size = port["size"]

            # Generate data_array signal declarations for 2d input port with bitwidth > 0
            if bitwidth > 0:
                port_decls.append(
                    f"    {name} : {dir} data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0)")

            # Use std_logic_vector for valid/ready of 2d input port
            port_decls.append(
                f"    {name}_valid : {dir} std_logic_vector({size} - 1 downto 0)")
            port_decls.append(
                f"    {name}_ready : {ready_dir} std_logic_vector({size} - 1 downto 0)")

            # Generate extra signal declarations for each item in the 2d input port
            for i in range(size):
                # Use the same extra signals for all items
                current_extra_signals = extra_signals

                # The netlist generator declares extra signals independently for each item,
                # in contrast to ready/valid signals.
                for signal_name, signal_bitwidth in current_extra_signals.items():
                    port_decls.append(
                        f"    {name}_{i}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")

    port_decls_str = ";\n".join(port_decls).lstrip()

    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity {entity_name} is
  port(
    clk : in std_logic;
    rst : in std_logic;
    {port_decls_str}
  );
end entity;
"""


def _get_default_extra_signal_value(extra_signal_name: str):
    return "\"0\""


def _get_forwarded_expression(signal_name: str, in_extra_signals: list[str]) -> str:
    if signal_name == "spec":
        return " or ".join(in_extra_signals)

    raise ValueError(
        f"Unsupported forwarding method for extra signal: {signal_name}")


def _forward_extra_signals(extra_signals: dict[str, int], in_ports) -> dict[str, str]:
    """
    Calculate how each extra signal is forwarded to the output ports.
    Result is a dict of extra signal names to VHDL expressions.
    e.g., {"spec": "lhs_spec or rhs_spec", "tag0": "lhs_tag0 (op) rhs_tag0"}
    If no inputs are provided, we use the default values.
    e.g., {"spec": "\"0\"", "tag0": "\"0\""}
    """

    forwarded_extra_signals: dict[str, str] = {}
    # Calculate forwarded extra signals
    for signal_name in extra_signals:
        in_extra_signals = []

        if not in_ports:
            # Use default values for extra signals
            forwarded_extra_signals[signal_name] = _get_default_extra_signal_value(
                signal_name)
        else:
            # Collect extra signals from all input ports
            for in_port in in_ports:
                port_name = in_port["name"]
                in_extra_signals.append(f"{port_name}_{signal_name}")

            # Forward all input extra signals with the specified method
            forwarded_extra_signals[signal_name] = _get_forwarded_expression(
                signal_name, in_extra_signals)

    return forwarded_extra_signals


def generate_inner_port_forwarding(ports) -> str:
    """
    Generate port forwarding for inner entity
    e.g.,
        lhs => lhs,
        lhs_valid => lhs_valid,
        lhs_ready => lhs_ready
    """
    forwardings = []
    for port in ports:
        port_name = port["name"]
        bitwidth = port["bitwidth"]

        # Forward data if present
        if bitwidth > 0:
            forwardings.append(f"      {port_name} => {port_name}")

        forwardings.append(f"      {port_name}_valid => {port_name}_valid")
        forwardings.append(f"      {port_name}_ready => {port_name}_ready")

    return ",\n".join(forwardings).lstrip()


def _generate_normal_signal_assignments(in_ports, out_ports, extra_signals) -> str:
    """
    e.g., result_spec <= lhs_spec or rhs_spec;
    """
    forwarded_extra_signals = _forward_extra_signals(
        extra_signals, in_ports)

    # Assign all extra signals for each output port, based on forwarded_extra_signals.
    # e.g., result_spec <= lhs_spec or rhs_spec;
    extra_signal_assignments = []
    for out_port in out_ports:
        port_name = out_port["name"]
        port_2d = out_port.get("2d", False)

        if not port_2d:
            # Assign all extra signals to this output port
            for signal_name in extra_signals:
                extra_signal_assignments.append(
                    f"  {port_name}_{signal_name} <= {forwarded_extra_signals[signal_name]};")
        else:
            port_size = out_port["size"]
            for signal_name in extra_signals:
                for i in range(port_size):
                    extra_signal_assignments.append(
                        f"  {port_name}_{i}_{signal_name} <= {forwarded_extra_signals[signal_name]};")

    return "\n".join(extra_signal_assignments).lstrip()


def _generate_normal_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]) -> str:
    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    entity = generate_entity(name, in_ports, out_ports)

    extra_signal_assignments = _generate_normal_signal_assignments(
        in_ports, out_ports, extra_signals)

    inner_port_forwarding = generate_inner_port_forwarding(in_ports + out_ports)

    architecture = f"""
-- Architecture of signal manager (normal)
architecture arch of {name} is
begin
  -- Forward extra signals to output ports
  {extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {inner_port_forwarding}
    );
end architecture;
"""

    return inner + entity + architecture


def _generate_buffered_transfer_logic(in_ports, out_ports):
    first_in_port_name = in_ports[0]["name"]
    first_out_port_name = out_ports[0]["name"]

    return f"""
  transfer_in <= {first_in_port_name}_valid and {first_in_port_name}_ready;
  transfer_out <= {first_out_port_name}_valid and {first_out_port_name}_ready;""".lstrip()


def _generate_buffered_signal_assignments(in_ports, out_ports, concat_info, extra_signals) -> str:
    """
    e.g., buff_in(0 downto 0) <= lhs_spec or rhs_spec;
    """
    forwarded_extra_signals = _forward_extra_signals(
        extra_signals, in_ports)

    # Concat/split extra signals for buffer input/output.
    signal_assignments = []

    # Generate assignments from individual extra signals to single concatenated variable.
    for signal_name, (msb, lsb) in concat_info.mapping:
        # Concat extra signals for buffer input.
        signal_assignments.append(
            f"  buff_in({msb} downto {lsb}) <= {forwarded_extra_signals[signal_name]};")

        # Assign extra signals to all output ports
        for out_port in out_ports:
            port_name = out_port["name"]

            # Split extra signals from buffer output.
            signal_assignments.append(
                f"  {port_name}_{signal_name} <= buff_out({msb} downto {lsb});")

    return "\n".join(signal_assignments).lstrip()


def _generate_buffered_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str], latency: int):
    # Delayed import to avoid circular dependency
    from generators.handshake.ofifo import generate_ofifo

    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    entity = generate_entity(name, in_ports, out_ports)

    # Get concatenation details for extra signals
    concat_info = ConcatenationInfo(extra_signals)
    extra_signals_bitwidth = concat_info.total_bitwidth

    # Generate buffer to store (concatenated) extra signals
    buff_name = f"{name}_buff"
    buff = generate_ofifo(buff_name, {
        "num_slots": latency,
        "bitwidth": extra_signals_bitwidth
    })

    # Generate transfer logic
    transfer_logic = _generate_buffered_transfer_logic(in_ports, out_ports)

    signal_assignments = _generate_buffered_signal_assignments(
        in_ports, out_ports, concat_info, extra_signals)

    forwarding = generate_inner_port_forwarding(in_ports + out_ports)

    architecture = f"""
-- Architecture of signal manager (buffered)
architecture arch of {name} is
  signal buff_in, buff_out : std_logic_vector({extra_signals_bitwidth} - 1 downto 0);
  signal transfer_in, transfer_out : std_logic;
begin
  -- Transfer signal assignments
  {transfer_logic}

  -- Concat/split extra signals for buffer input/output
  {signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {forwarding}
    );

  -- Generate ofifo to store extra signals
  -- num_slots = {latency}, bitwidth = {extra_signals_bitwidth}
  buff : entity work.{buff_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => buff_in,
      ins_valid => transfer_in,
      ins_ready => open,
      outs => buff_out,
      outs_valid => open,
      outs_ready => transfer_out
    );
end architecture;
"""

    return inner + buff + entity + architecture


class ConcatenationInfo:
    # List of tuples of (extra_signal_name, (msb, lsb))
    # e.g., [("spec", (0, 0)), ("tag0", (8, 1))]
    mapping: list[tuple[str, tuple[int, int]]]
    total_bitwidth: int

    def __init__(self, extra_signals: dict[str, int]):
        self.mapping = []
        self.total_bitwidth = 0

        for name, bitwidth in extra_signals.items():
            self.add(name, bitwidth)

    def add(self, name: str, bitwidth: int):
        self.mapping.append(
            (name, (self.total_bitwidth + bitwidth - 1, self.total_bitwidth)))
        self.total_bitwidth += bitwidth

    def has(self, name: str) -> bool:
        return name in [name for name, _ in self.mapping]

    def get(self, name: str):
        return self.mapping[[name for name, _ in self.mapping].index(name)]


def get_concat_extra_signals_bitwidth(extra_signals: dict[str, int]):
    return sum(extra_signals.values())


def generate_concat_signal_decls(ports, extra_signals_bitwidth, ignore=[]) -> str:
    """
    Declare signals for concatenated data and extra signals
    e.g., signal lhs_inner : std_logic_vector(33 - 1 downto 0); // 32 (data) + 1 (spec)
    """
    signal_decls = []
    for port in ports:
        port_name = port["name"]
        port_bitwidth = port["bitwidth"]
        port_2d = port.get("2d", False)

        # Ignore some ports
        if port_name in ignore:
            continue

        # Concatenated bitwidth
        full_bitwidth = extra_signals_bitwidth + port_bitwidth

        if port_2d:
            port_size = port["size"]

            # Inner signal is data_array
            signal_decls.append(
                f"  signal {port_name}_inner : data_array({port_size} - 1 downto 0)({full_bitwidth} - 1 downto 0);")
        else:
            signal_decls.append(
                f"  signal {port_name}_inner : std_logic_vector({full_bitwidth} - 1 downto 0);")

    return "\n".join(signal_decls).lstrip()


def generate_concat_logic(in_ports, out_ports, concat_info, ignore=[]):
    """
    Generate concat logic for all input/output ports
    e.g.,
    lhs_inner(31 downto 0) <= lhs;
    lhs_inner(32 downto 32) <= lhs_spec;
    ...
    result <= result_inner(31 downto 0);
    result_spec <= result_inner(32 downto 32);
    """
    concat_logic = []
    for port in in_ports:
        port_name = port["name"]
        port_bitwidth = port["bitwidth"]
        port_2d = port.get("2d", False)

        # Ignore some ports
        if port_name in ignore:
            continue

        if port_2d:
            port_size = port["size"]
            for i in range(port_size):
                # Include data if present
                if port_bitwidth > 0:
                    concat_logic.append(
                        f"  {port_name}_inner({i})({port_bitwidth} - 1 downto 0) <= {port_name}({i});")

                # Include all extra signals
                for signal_name, (msb, lsb) in concat_info.mapping:
                    concat_logic.append(
                        f"  {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{i}_{signal_name};")
        else:
            # Include data if present
            if port_bitwidth > 0:
                concat_logic.append(
                    f"  {port_name}_inner({port_bitwidth} - 1 downto 0) <= {port_name};")

            # Include all extra signals
            for signal_name, (msb, lsb) in concat_info.mapping:
                concat_logic.append(
                    f"  {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{signal_name};")

    for port in out_ports:
        port_name = port["name"]
        port_bitwidth = port["bitwidth"]
        port_2d = port.get("2d", False)

        # Ignore some ports
        if port_name in ignore:
            continue

        if port_2d:
            port_size = port["size"]
            for i in range(port_size):
                # Extract data if present
                if port_bitwidth > 0:
                    concat_logic.append(
                        f"  {port_name}({i}) <= {port_name}_inner({i})({port_bitwidth} - 1 downto 0);")

                # Extract all extra signals
                for signal_name, (msb, lsb) in concat_info.mapping:
                    concat_logic.append(
                        f"  {port_name}_{i}_{signal_name} <= {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth});")
        else:
            # Extract data if present
            if port_bitwidth > 0:
                concat_logic.append(
                    f"  {port_name} <= {port_name}_inner({port_bitwidth} - 1 downto 0);")

            # Extract all extra signals
            for signal_name, (msb, lsb) in concat_info.mapping:
                concat_logic.append(
                    f"  {port_name}_{signal_name} <= {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth});")

    return "\n".join(concat_logic).lstrip()


def _generate_concat_forwarding(in_ports, out_ports, handled_extra_signals, ignore_ports) -> str:
    """
    Port forwarding for the inner entity of concat signal manager
    We can't use `_generate_inner_port_forwarding()` because:
    (1) Data is always forwarded, regardless of (port's original) bitwidth, due to the concatenation.
    (2) Data ports must be renamed to `_inner`.
    e.g., lhs => lhs_inner,
          lhs_valid => lhs_valid,
          lhs_ready => lhs_ready
    """

    forwardings = []
    for port in in_ports + out_ports:
        port_name = port["name"]

        if port["name"] in ignore_ports:
            # Forward the original data signal, because it's not concatenated
            forwardings.append(f"      {port_name} => {port_name}")
        else:
            forwardings.append(f"      {port_name} => {port_name}_inner")

        forwardings.append(f"      {port_name}_valid => {port_name}_valid")
        forwardings.append(f"      {port_name}_ready => {port_name}_ready")

        # Forward unhandled extra signals
        for signal in port.get("extra_signals", {}):
            if signal not in handled_extra_signals:
                forwardings.append(
                    f"      {port_name}_{signal} => {port_name}_{signal}")

    return ",\n".join(forwardings).lstrip()


def _generate_concat_signal_manager(name, in_ports, out_ports, extra_signals, ignore_ports, generate_inner: Callable[[str], str]):
    entity = generate_entity(name, in_ports, out_ports)

    # Exclude specified ports for concatenation
    filtered_in_ports = [
        port for port in in_ports if not port["name"] in ignore_ports]
    filtered_out_ports = [
        port for port in out_ports if not port["name"] in ignore_ports]

    # Get concatenation details for extra signals
    concat_info = ConcatenationInfo(extra_signals)
    extra_signals_bitwidth = concat_info.total_bitwidth

    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    # Declare inner concatenated signals for all input/output ports
    concat_signal_decls = generate_concat_signal_decls(
        filtered_in_ports + filtered_out_ports, extra_signals_bitwidth)

    # Assign inner concatenated signals
    concat_logic = generate_concat_logic(
        filtered_in_ports, filtered_out_ports, concat_info)

    # Port forwarding for the inner entity
    forwardings = _generate_concat_forwarding(
        in_ports, out_ports, extra_signals, ignore_ports)

    architecture = f"""
-- Architecture of signal manager (concat)
architecture arch of {name} is
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Concatenate data and extra signals
  {concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {forwardings}
    );
end architecture;
"""

    return inner + entity + architecture


def _generate_bbmerge_index_extra_signal_assignments(index_name, index_extra_signals, index_dir) -> str:
    """
    e.g., index_tag0 <= "0";
    """
    # TODO: Extra signals for index port are not tested
    if index_dir == "out" and index_extra_signals:
        index_extra_signals_list = []
        for signal_name in index_extra_signals:
            index_extra_signals_list.append(
                f"  {index_name}_{signal_name} <= {_get_default_extra_signal_value(signal_name)};")
        return "\n".join(index_extra_signals_list)
    return ""


def _generate_bbmerge_signal_assignments(concat_logic, index_extra_signal_assignments) -> str:
    template = f"""
  -- Concatenate data and extra signals
  {concat_logic}
"""

    if index_extra_signal_assignments:
        template += f"""
  -- Assign index extra signals
  {index_extra_signal_assignments}
"""

    return template.lstrip()


def _generate_bbmerge_signal_manager(name, in_ports, out_ports, index_name, extra_signals, index_dir, generate_inner: Callable[[str], str]):
    entity = generate_entity(name, in_ports, out_ports)

    # Get concatenation details for extra signals
    concat_info = ConcatenationInfo(extra_signals)
    extra_signals_bitwidth = concat_info.total_bitwidth

    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    # Declare inner concatenated signals for all input/output ports
    concat_signal_decls = generate_concat_signal_decls(
        in_ports + out_ports, extra_signals_bitwidth, ignore=[index_name])

    # Assign inner concatenated signals
    concat_logic = generate_concat_logic(
        in_ports, out_ports, concat_info, ignore=[index_name])

    # Assign index extra signals
    index_extra_signal_assignments = _generate_bbmerge_index_extra_signal_assignments(
        index_name, extra_signals, index_dir)

    signal_assignments = _generate_bbmerge_signal_assignments(
        concat_logic, index_extra_signal_assignments)

    # Port forwarding for the inner entity
    forwardings = _generate_concat_forwarding(
        in_ports, out_ports, extra_signals, [index_name])

    architecture = f"""
-- Architecture of signal manager (bbmerge)
architecture arch of {name} is
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  {signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {forwardings}
    );
end architecture;
"""

    return inner + entity + architecture

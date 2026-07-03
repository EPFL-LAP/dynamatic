from generators.support.utils import ExtraSignals


def generate_ii_monitor(name, params):
    # Bit-width of the dominating control_merge's index output data signal.
    index_width = params["index_width"]
    # Index value identifying a back-edge (loop-internal) activation.
    loop_back_index = params["loop_back_index"]
    # Nesting depth of the measured loop (1 for a top-level loop) and deepest
    # depth reachable from it through its own descendants (equal to
    # loop_depth when the loop is innermost).
    loop_depth = params["loop_depth"]
    loop_max_depth = params["loop_max_depth"]
    # Extra signals carried by the observed index/exit channels (e.g. a "spec"
    # signal under speculation). The monitor ignores their values but must
    # declare matching ports so that its instantiation in the parent module is
    # well-formed.
    index_extra_signals = params.get("index_extra_signals", None) or {}
    exit_extra_signals = params.get("exit_extra_signals", None) or {}

    return _generate_ii_monitor(name, index_width, loop_back_index,
                                loop_depth, loop_max_depth,
                                index_extra_signals, exit_extra_signals)


def _extra_signal_ports(channel_name: str,
                        extra_signals: ExtraSignals) -> list[str]:
    return [
        f"{channel_name}_{signal_name} : in std_logic_vector({signal_width} - 1 downto 0)"
        for signal_name, signal_width in extra_signals.items()
    ]


def _generate_ii_monitor(name: str, index_width: int, loop_back_index: int,
                         loop_depth: int, loop_max_depth: int,
                         index_extra_signals: ExtraSignals,
                         exit_extra_signals: ExtraSignals) -> str:
    # Assemble the port list. All ports are inputs so the monitor never drives
    # the circuit (it only taps existing wires).
    ports = [
        "clk         : in std_logic",
        "rst         : in std_logic",
        # Index channel of the dominating control_merge (read-only).
        f"index       : in std_logic_vector({index_width} - 1 downto 0)",
        "index_valid : in std_logic",
        "index_ready : in std_logic",
        *_extra_signal_ports("index", index_extra_signals),
        # An exit channel of the loop (read-only).
        "exit_valid  : in std_logic",
        "exit_ready  : in std_logic",
        *_extra_signal_ports("exit", exit_extra_signals),
    ]
    port_decls = ";\n    ".join(ports)

    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- II monitor: simulation-only component that measures the initiation interval
-- (II) of a loop at nesting depth {loop_depth} (of {loop_max_depth} in its
-- nest) by observing its dominating control_merge's index channel.
--
-- All ports are inputs so that the monitor passively reads existing signals
-- without driving any of them, avoiding multiple-driver conflicts with the
-- actual circuit. Any extra signals on the observed channels are declared (so
-- the instantiation matches) but left unused.
--
-- Measurement logic:
--   * When the control_merge fires (index_valid and index_ready):
--       - index /= {loop_back_index} -> activation from outside the loop (entry/re-entry):
--           report the previous window (if more than one iteration was seen)
--           and start a fresh measurement window.
--       - index  = {loop_back_index} -> back-edge activation: advance the window.
--   * When the exit channel fires (exit_valid and exit_ready):
--       report the current window (if more than one iteration was seen) and
--       stop measuring.
entity {name} is
  port (
    {port_decls}
  );
end entity;

architecture arch of {name} is
begin
  process (clk)
    variable cycle     : integer := 0;
    variable measuring : boolean := false;
    variable start_cyc : integer := 0;
    variable last_cyc  : integer := 0;
    variable iters     : integer := 0;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        cycle     := 0;
        measuring := false;
        start_cyc := 0;
        last_cyc  := 0;
        iters     := 0;
      else
        cycle := cycle + 1;

        -- Observe the control_merge index channel.
        if index_valid = '1' and index_ready = '1' then
          if to_integer(unsigned(index)) /= {loop_back_index} then
            -- Activation from outside the loop (first entry or re-entry).
            if measuring and iters > 1 then
              report "II_INSTRUMENT: loop=" & {name}'path_name &
                     " depth={loop_depth}/{loop_max_depth} II=" &
                     real'image(real(last_cyc - start_cyc) / real(iters - 1)) &
                     " iterations=" & integer'image(iters) severity note;
            end if;
            measuring := true;
            start_cyc := cycle;
            last_cyc  := cycle;
            iters     := 1;
          else
            -- Activation from the loop back-edge.
            if measuring then
              last_cyc := cycle;
              iters    := iters + 1;
            end if;
          end if;
        end if;

        -- Observe the loop exit channel.
        if exit_valid = '1' and exit_ready = '1' then
          if measuring and iters > 1 then
            report "II_INSTRUMENT: loop=" & {name}'path_name &
                   " depth={loop_depth}/{loop_max_depth} II=" &
                   real'image(real(last_cyc - start_cyc) / real(iters - 1)) &
                   " iterations=" & integer'image(iters) severity note;
          end if;
          measuring := false;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

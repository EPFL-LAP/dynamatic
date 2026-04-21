from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_counter_buffer(name, params):
    bitwidth = params["bitwidth"]
    dv_latency = int(params["dv_latency"])
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_counter_buffer_signal_manager(name, dv_latency, bitwidth, extra_signals)
    if bitwidth == 0:
        return _generate_counter_buffer_dataless(name, dv_latency)
    else:
        return _generate_counter_buffer(name, dv_latency, bitwidth)


def _counter_width(dv_latency):
    return 1 if dv_latency <= 1 else (dv_latency - 1).bit_length()


def _generate_counter_buffer_dataless(name, dv_latency):
    cnt_w = _counter_width(dv_latency)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of counter_buffer_dataless
entity {name} is
  port(
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of counter_buffer_dataless
architecture arch of {name} is
  signal occupied : std_logic;
  signal delayCnt : unsigned({cnt_w} - 1 downto 0);
  signal done     : std_logic;
begin
  done <= '1' when (occupied = '1' and delayCnt = to_unsigned(0, {cnt_w})) else '0';

  outs_valid <= done;
  ins_ready  <= (not occupied) or (done and outs_ready);

  process (clk) is
  begin
    if rising_edge(clk) then
      if rst = '1' then
        occupied <= '0';
        delayCnt <= (others => '0');
      elsif occupied = '0' then
        if ins_valid = '1' then
          occupied <= '1';
          delayCnt <= to_unsigned({dv_latency - 1}, {cnt_w});
        end if;
      elsif delayCnt > to_unsigned(0, {cnt_w}) then
        delayCnt <= delayCnt - 1;
      elsif outs_ready = '1' then
        if ins_valid = '1' then
          delayCnt <= to_unsigned({dv_latency - 1}, {cnt_w});
        else
          occupied <= '0';
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_counter_buffer(name, dv_latency, bitwidth):
    inner_name = f"{name}_inner"
    dependencies = _generate_counter_buffer_dataless(inner_name, dv_latency)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of counter_buffer
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of counter_buffer
architecture arch of {name} is
  signal inputReady : std_logic;
  signal loadData   : std_logic;
begin

  control : entity work.{inner_name}
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  loadData <= ins_valid and inputReady;
  ins_ready <= inputReady;

  process (clk) is
  begin
    if rising_edge(clk) then
      if rst = '1' then
        outs <= (others => '0');
      elsif loadData = '1' then
        outs <= ins;
      end if;
    end if;
  end process;
end architecture;
"""

    return dependencies + entity + architecture


def _generate_counter_buffer_signal_manager(name, dv_latency, bitwidth, extra_signals):
    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
    return generate_concat_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda inner_name: _generate_counter_buffer(
            inner_name,
            dv_latency,
            bitwidth + extra_signals_bitwidth
        ))

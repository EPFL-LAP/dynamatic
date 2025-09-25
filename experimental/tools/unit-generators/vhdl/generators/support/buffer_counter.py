def generate_buffer_counter(name, slots):
    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

-- Entity of buffer_counter
entity {name} is
  port (
    clk, rst : in std_logic;
    ins_valid : in std_logic;
    ins_ready : in std_logic;
    outs_valid : in std_logic;
    outs_ready : in std_logic
  );
end entity;

-- Architecture of buffer_counter
architecture arch of {name} is
  constant counter_width : integer := integer(ceil(log2(real(1 + {slots}))));
  signal counter : std_logic_vector(counter_width - 1 downto 0);
  signal error : std_logic;
  signal write_en, read_en : std_logic;
begin
  write_en <= ins_valid and ins_ready;
  read_en <= outs_valid and outs_ready;

  counter_proc : process(clk, rst)
  begin
    if rst = '1' then
      counter <= "0";
    elsif rising_edge(clk) then
      if write_en = '1' and read_en = '1' then
        counter <= counter;
      elsif write_en = '1' and unsigned(counter) < {slots} then
        counter <= std_logic_vector(unsigned(counter) + 1);
      elsif read_en = '1' and unsigned(counter) > 0 then
        counter <= std_logic_vector(unsigned(counter) - 1);
      end if;
    end if;
  end process;
  error_proc : process(clk, rst)
  begin
    if rst = '1' then
      error <= '0';
    elsif rising_edge(clk) then
      if write_en = '1' and read_en = '1' then
        error <= error;
      elsif write_en = '1' and unsigned(counter) = {slots} then
        error <= '1';
      elsif read_en = '1' and unsigned(counter) = 0 then
        error <= '1';
      end if;
    end if;
  end process;
end architecture;
"""


def generate_buffer_counter_embedding(name):
    return f"""
  debug_counter : entity work.{name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
""".lstrip()

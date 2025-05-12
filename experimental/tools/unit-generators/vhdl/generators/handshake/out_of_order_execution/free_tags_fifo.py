from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner


def generate_free_tags_fifo(name, params):
  bitwidth = params["bitwidth"]
  fifo_depth = params["fifo_depth"]

  return _generate_free_tags_fifo(name, bitwidth, fifo_depth)


def _generate_free_tags_fifo(name, bitwidth, fifo_depth):
  fifo_name = f"{name}_fifo"

  dependencies = \
      generate_elastic_fifo_inner(fifo_name, {
          "size": fifo_depth,
          "bitwidth": bitwidth,
          "initialized": True
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of free_tags_fifo
entity {name} is 
port (
        clk, rst      : in  std_logic;
        ins           : in  std_logic_vector({bitwidth} - 1 downto 0);
        outs          : out std_logic_vector({bitwidth} - 1 downto 0);
        ins_valid     : in  std_logic;
        outs_ready    : in  std_logic;
        outs_valid    : out std_logic;
        ins_ready     : out std_logic
);
end entity;
"""
  architecture = f"""
-- Architecture of free_tags_fifo
architecture arch of {name} is
    signal mux_sel : std_logic;
    signal fifo_valid, fifo_ready : STD_LOGIC;
    signal fifo_pvalid, fifo_nready : STD_LOGIC;
    signal fifo_in, fifo_out: std_logic_vector({bitwidth}-1 downto 0);
begin
    
    process (mux_sel, fifo_out, ins) is
        begin
            if (mux_sel = '1') then
                outs <= fifo_out;
            else
                outs <= ins;
            end if;
    end process;

    outs_valid <= ins_valid or fifo_valid;    --fifo_valid is 0 only if fifo is empty
    ins_ready <= fifo_ready or outs_ready;
    fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid); --store in FIFO if next is not ins_ready or FIFO is already outputting something
    mux_sel <= fifo_valid;

    fifo_nready <= outs_ready;
    fifo_in <= ins;

    fifo: entity work.{fifo_name}(arch)
        port map (
        --inputs
            clk => clk, 
            rst => rst,
            ins =>fifo_in, 
            ins_valid  => fifo_pvalid, 
            outs_ready => fifo_nready,    
            
        --outputs
            outs => fifo_out,
            outs_valid => fifo_valid,
            ins_ready => fifo_ready   
        );
end architecture;
"""
  return dependencies + entity + architecture

from generators.support.utils import VhdlScalarType
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner

def generate_free_tags_fifo(name, params):
  fifo_depth = params["fifo_depth"]
  port_types = params["port_types"]
  bitwidth = VhdlScalarType(port_types["outs"]).bitwidth

  return _generate_free_tags_fifo(name, bitwidth, fifo_depth)

def _generate_free_tags_fifo(name, bitwidth, fifo_depth):
  fifo_name = f"{name}_fifo"

  dependencies = \
      generate_elastic_fifo_inner(fifo_name, {
          "size": fifo_depth,
          "bitwidth": bitwidth
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity {name} is 
port (
        clk, rst      : in  std_logic;
        dataIn        : in  std_logic_vector({bitwidth} - 1 downto 0);
        dataOut       : out std_logic_vector({bitwidth} - 1 downto 0);
        pValid        : in  std_logic;
        nReady        : in  std_logic;
        valid         : out std_logic;
        ready         : out std_logic
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
    
    process (mux_sel, fifo_out, dataIn) is
        begin
            if (mux_sel = '1') then
                dataOut <= fifo_out;
            else
                dataOut <= dataIn;
            end if;
    end process;

    valid <= pValid or fifo_valid;    --fifo_valid is 0 only if fifo is empty
    ready <= fifo_ready or nReady;
    fifo_pvalid <= pValid and (not nReady or fifo_valid); --store in FIFO if next is not ready or FIFO is already outputting something
    mux_sel <= fifo_valid;

    fifo_nready <= nReady;
    fifo_in <= dataIn;

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
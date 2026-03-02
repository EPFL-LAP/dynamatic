-- tagger : tagger({'data_bitwidth': 32, 'tag_bitwidth': 1, 'input_extra_signals': {}, 'output_extra_signals': {'tag': 1}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity tagger_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of tagger_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity tagger_join is
  port (
    -- inputs
    ins_valid  : in std_logic_vector(2 - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of join
architecture arch of tagger_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.tagger_join_and_n port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(2 - 1 downto 0);
  begin
    for i in 0 to 2 - 1 loop
      singlePValid(i) := '1';
      for j in 0 to 2 - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to 2 - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

-- Entity of tagger
entity tagger is
  port(
    clk        : in std_logic;
    rst        : in std_logic;
    ins_valid : in std_logic;
    
    outs_ready : in std_logic; 
    outs_valid : out std_logic;

    ins_ready : out std_logic;

    ins   : in  std_logic_vector(32 - 1 downto 0);
    outs  : out std_logic_vector(32 - 1 downto 0);

    dataIn : in std_logic_vector(1-1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_ready : out std_logic;

    outs_tag : out std_logic_vector(1-1 downto 0) 
  );
end entity;

-- Architecture of tagger
architecture arch of tagger is
  signal combined_valid : std_logic_vector(1 downto 0);
  signal combined_ready : std_logic_vector(1 downto 0);
begin
    -- Combine dataIn_valid and ins_valid
    combined_valid <= dataIn_valid & ins_valid;

    j : entity work.tagger_join
                port map(   combined_valid,
                            outs_ready,
                            outs_valid,
                            combined_ready);

    outs <= ins;

    -- Split combined_ready into ins_ready and dataIn_ready
    ins_ready   <= combined_ready(0);
    dataIn_ready <= combined_ready(1);

    outs_tag <= dataIn;

end architecture;


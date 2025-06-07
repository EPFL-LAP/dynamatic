-----------------------------------------------------------------------
-- FloatingPointAdder, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

entity addf_32_2_705000 is
Generic (
 INPUTS: integer := 2; 
 OUTPUTS: integer := 1; 
 DATA_SIZE_IN: integer := 32; --we default to single precision
 DATA_SIZE_OUT: integer := 32
);
port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
     lhs          : in std_logic_vector(31 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(31  downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
     result       : out std_logic_vector(31  downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of addf_32_2_705000 is
    
    
    -- legacy comment : main_component went here in component based version

    signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);

    --intermediate input signals for float conversion
    signal ip_lhs, ip_rhs : std_logic_vector(33 downto 0);

    --intermidiate output signal(s) for float conversion
    signal ip_result : std_logic_vector(33 downto 0);

    

    begin


          join_inputs : entity work.join(arch) generic map(2) 
    port map( 
      -- inputs 
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs 
      outs_valid   => join_valid, 
      ins_ready(0) => lhs_ready, 
      ins_ready(1) => rhs_ready
    );

        buff: entity work.delay_buffer(arch) generic map(9)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.oehb_dataless(arch)
            port map(
            clk        => clk,
            rst        => rst,
            ins_valid  => buff_valid,
            outs_ready => result_ready,
            outs_valid => result_valid,
            ins_ready  => oehb_ready
            );

        ieee2nfloat_0: entity work.InputIEEE_32bit(arch)
                port map (
                    --input
                    X =>lhs,
                    --output
                    R => ip_lhs
                );

        ieee2nfloat_1: entity work.InputIEEE_32bit(arch)
                port map (
                    --input
                    X => rhs,
                    --output
                    R => ip_rhs
                );

        

        nfloat2ieee : entity work.OutputIEEE_32bit(arch)
                port map (
                    --input
                    X => ip_result,
                    --ouput
                    R => result
                );

        operator : entity work.FloatingPointAdder_32_2_705(arch)
        port map (
            clk   => clk,
            ce_1 => oehb_ready,
            ce_2 => oehb_ready,
            ce_3 => oehb_ready,
            ce_4 => oehb_ready,
            ce_5 => oehb_ready,
            ce_6 => oehb_ready,
            ce_7 => oehb_ready,
            ce_8 => oehb_ready,
            ce_9 => oehb_ready,
            ce_10 => oehb_ready,
            X     => ip_lhs,
            Y     => ip_rhs,
            R     => ip_result
        );
end architecture;




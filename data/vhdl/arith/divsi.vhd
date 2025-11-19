library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity divsi is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of divsi is

  -- Interface to Vivado component
	component dynamatic_units_sdiv_32ns_32ns_32_36_1 is
		generic (
				  ID : INTEGER;
				  NUM_STAGE : INTEGER;
				  din0_WIDTH : INTEGER;
				  din1_WIDTH : INTEGER;
				  dout_WIDTH : INTEGER);
		port (
			     clk : IN STD_LOGIC;
			     reset : IN STD_LOGIC;
			     ce : IN STD_LOGIC;
			     din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
			     din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
			     dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
	end component;

  signal join_valid : std_logic;
  signal done : std_logic;
	constant LATENCY : integer := 35;

  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic;

begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );


	ip : component dynamatic_units_sdiv_32ns_32ns_32_36_1
	generic map (
				ID => 1,
				NUM_STAGE => LATENCY + 1,
				din0_WIDTH => 32,
				din1_WIDTH => 32,
				dout_WIDTH => 32)
	port map (
			   clk => clk,
			   reset => rst,
			   ce => oehb_ready,
			   din0 => lhs,
			   din1 => rhs,
			   dout => result);

  buff : entity work.delay_buffer(arch) generic map(34)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.oehb(arch) generic map(1)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins(0)     => oehb_datain,
      outs(0)    => oehb_dataOut
    );

end architecture;


-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1.1 (64-bit)
-- Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dynamatic_units_sdiv_32ns_32ns_32_36_1_div_u is
    generic (
        in0_WIDTH   : INTEGER :=32;
        in1_WIDTH   : INTEGER :=32;
        out_WIDTH   : INTEGER :=32);
    port (
        clk         : in  STD_LOGIC;
        reset       : in  STD_LOGIC;
        ce          : in  STD_LOGIC;
        dividend    : in  STD_LOGIC_VECTOR(in0_WIDTH-1 downto 0);
        divisor     : in  STD_LOGIC_VECTOR(in1_WIDTH-1 downto 0);
        sign_i      : in  STD_LOGIC_VECTOR(1 downto 0);
        sign_o      : out STD_LOGIC_VECTOR(1 downto 0);
        quot        : out STD_LOGIC_VECTOR(out_WIDTH-1 downto 0);
        remd        : out STD_LOGIC_VECTOR(out_WIDTH-1 downto 0));

    function max (left, right : INTEGER) return INTEGER is
    begin
        if left > right then return left;
        else return right;
        end if;
    end max;

end entity;

architecture rtl of dynamatic_units_sdiv_32ns_32ns_32_36_1_div_u is
    constant cal_WIDTH      : INTEGER := max(in0_WIDTH, in1_WIDTH);
    type  in0_vector  is array(INTEGER range <>) of UNSIGNED(in0_WIDTH-1 downto 0);
    type  in1_vector  is array(INTEGER range <>) of UNSIGNED(in1_WIDTH-1 downto 0);
    type  cal_vector  is array(INTEGER range <>) of UNSIGNED(cal_WIDTH downto 0);
    type  sign_vector is array(INTEGER range <>) of UNSIGNED(1 downto 0);

    signal dividend_tmp     : in0_vector(0 to in0_WIDTH);
    signal divisor_tmp      : in1_vector(0 to in0_WIDTH);
    signal remd_tmp         : in0_vector(0 to in0_WIDTH);
    signal comb_tmp         : in0_vector(0 to in0_WIDTH-1);
    signal cal_tmp          : cal_vector(0 to in0_WIDTH-1);
    signal sign_tmp         : sign_vector(0 to in0_WIDTH);
begin
    quot   <= STD_LOGIC_VECTOR(RESIZE(dividend_tmp(in0_WIDTH), out_WIDTH));
    remd   <= STD_LOGIC_VECTOR(RESIZE(remd_tmp(in0_WIDTH), out_WIDTH));
    sign_o <= STD_LOGIC_VECTOR(sign_tmp(in0_WIDTH));

    tran_tmp_proc : process (clk)
    begin
        if (clk'event and clk='1') then
            if (ce = '1') then
                dividend_tmp(0) <= UNSIGNED(dividend);
                divisor_tmp(0)  <= UNSIGNED(divisor);
                sign_tmp(0) <= UNSIGNED(sign_i);
                remd_tmp(0) <= (others => '0');
            end if;
        end if;
    end process tran_tmp_proc;

    run_proc: for i in 0 to in0_WIDTH-1 generate
    begin
        comb_tmp(i) <= remd_tmp(i)(in0_WIDTH-2 downto 0) & dividend_tmp(i)(in0_WIDTH-1);
        cal_tmp(i)  <= ('0' & comb_tmp(i)) - ('0' & divisor_tmp(i));

        process (clk)
        begin
            if (clk'event and clk='1') then
                if (ce = '1') then
                    dividend_tmp(i+1) <= dividend_tmp(i)(in0_WIDTH-2 downto 0) & (not cal_tmp(i)(cal_WIDTH));
                    divisor_tmp(i+1)  <= divisor_tmp(i);
                    sign_tmp(i+1)     <= sign_tmp(i);
                    if cal_tmp(i)(cal_WIDTH) = '1' then
                        remd_tmp(i+1) <= comb_tmp(i);
                    else
                        remd_tmp(i+1) <= cal_tmp(i)(in0_WIDTH-1 downto 0);
                    end if;
                end if;
            end if;
        end process;
    end generate run_proc;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dynamatic_units_sdiv_32ns_32ns_32_36_1_div is
    generic (
        in0_WIDTH   : INTEGER :=32;
        in1_WIDTH   : INTEGER :=32;
        out_WIDTH   : INTEGER :=32);
    port (
        clk         : in  STD_LOGIC;
        reset       : in  STD_LOGIC;
        ce          : in  STD_LOGIC;
        dividend    : in  STD_LOGIC_VECTOR(in0_WIDTH-1 downto 0);
        divisor     : in  STD_LOGIC_VECTOR(in1_WIDTH-1 downto 0);
        quot        : out STD_LOGIC_VECTOR(out_WIDTH-1 downto 0);
        remd        : out STD_LOGIC_VECTOR(out_WIDTH-1 downto 0));
end entity;

architecture rtl of dynamatic_units_sdiv_32ns_32ns_32_36_1_div is
    component dynamatic_units_sdiv_32ns_32ns_32_36_1_div_u is
        generic (
            in0_WIDTH   : INTEGER :=32;
            in1_WIDTH   : INTEGER :=32;
            out_WIDTH   : INTEGER :=32);
        port (
            reset       : in  STD_LOGIC;
            clk         : in  STD_LOGIC;
            ce          : in  STD_LOGIC;
            dividend    : in  STD_LOGIC_VECTOR(in0_WIDTH-1 downto 0);
            divisor     : in  STD_LOGIC_VECTOR(in1_WIDTH-1 downto 0);
            sign_i      : in  STD_LOGIC_VECTOR(1 downto 0);
            sign_o      : out STD_LOGIC_VECTOR(1 downto 0);
            quot        : out STD_LOGIC_VECTOR(out_WIDTH-1 downto 0);
            remd        : out STD_LOGIC_VECTOR(out_WIDTH-1 downto 0));
    end component;

    signal dividend0  : STD_LOGIC_VECTOR(in0_WIDTH-1 downto 0);
    signal divisor0   : STD_LOGIC_VECTOR(in1_WIDTH-1 downto 0);
    signal dividend_u : STD_LOGIC_VECTOR(in0_WIDTH-1 downto 0);
    signal divisor_u  : STD_LOGIC_VECTOR(in1_WIDTH-1 downto 0);
    signal quot_u     : STD_LOGIC_VECTOR(out_WIDTH-1 downto 0);
    signal remd_u     : STD_LOGIC_VECTOR(out_WIDTH-1 downto 0);
    signal sign_i     : STD_LOGIC_VECTOR(1 downto 0);
    signal sign_o     : STD_LOGIC_VECTOR(1 downto 0);
begin
    dynamatic_units_sdiv_32ns_32ns_32_36_1_div_u_0 : dynamatic_units_sdiv_32ns_32ns_32_36_1_div_u
        generic map(
            in0_WIDTH   => in0_WIDTH,
            in1_WIDTH   => in1_WIDTH,
            out_WIDTH   => out_WIDTH)
        port map(
            clk         => clk,
            reset       => reset,
            ce          => ce,
            dividend    => dividend_u,
            divisor     => divisor_u,
            sign_i      => sign_i,
            sign_o      => sign_o,
            quot        => quot_u,
            remd        => remd_u);

    sign_i      <= (dividend0(in0_WIDTH-1) xor divisor0(in1_WIDTH-1)) & dividend0(in0_WIDTH-1);
    dividend_u  <= STD_LOGIC_VECTOR(UNSIGNED(not dividend0) + 1) when dividend0(in0_WIDTH-1) = '1' else dividend0;
    divisor_u   <= STD_LOGIC_VECTOR(UNSIGNED(not divisor0) + 1) when divisor0(in1_WIDTH-1) = '1' else divisor0;

process (clk)
begin
    if (clk'event and clk = '1') then
        if (ce = '1') then
            dividend0 <= dividend;
            divisor0 <= divisor;
        end if;
    end if;
end process;

process (clk)
begin
    if (clk'event and clk = '1') then
        if (ce = '1') then
            if (sign_o(1) = '1') then
                quot <= STD_LOGIC_VECTOR(UNSIGNED(not quot_u) + 1);
            else
                quot <= quot_u;
            end if;
        end if;
    end if;
end process;

process (clk)
begin
    if (clk'event and clk = '1') then
        if (ce = '1') then
            if (sign_o(0) = '1') then
                remd <= STD_LOGIC_VECTOR(UNSIGNED(not remd_u) + 1);
            else
                remd <= remd_u;
            end if;
        end if;
    end if;
end process;

end architecture;


Library IEEE;
use IEEE.std_logic_1164.all;

entity dynamatic_units_sdiv_32ns_32ns_32_36_1 is
    generic (
        ID : INTEGER;
        NUM_STAGE : INTEGER;
        din0_WIDTH : INTEGER;
        din1_WIDTH : INTEGER;
        dout_WIDTH : INTEGER);
    port (
        clk : IN STD_LOGIC;
        reset : IN STD_LOGIC;
        ce : IN STD_LOGIC;
        din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
        din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
        dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
end entity;

architecture arch of dynamatic_units_sdiv_32ns_32ns_32_36_1 is
    component dynamatic_units_sdiv_32ns_32ns_32_36_1_div is
        generic (
            in0_WIDTH : INTEGER;
            in1_WIDTH : INTEGER;
            out_WIDTH : INTEGER);
        port (
            dividend : IN STD_LOGIC_VECTOR;
            divisor : IN STD_LOGIC_VECTOR;
            quot : OUT STD_LOGIC_VECTOR;
            remd : OUT STD_LOGIC_VECTOR;
            clk : IN STD_LOGIC;
            ce : IN STD_LOGIC;
            reset : IN STD_LOGIC);
    end component;

    signal sig_quot : STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0);
    signal sig_remd : STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0);


begin
    dynamatic_units_sdiv_32ns_32ns_32_36_1_div_U :  component dynamatic_units_sdiv_32ns_32ns_32_36_1_div
    generic map (
        in0_WIDTH => din0_WIDTH,
        in1_WIDTH => din1_WIDTH,
        out_WIDTH => dout_WIDTH)
    port map (
        dividend => din0,
        divisor => din1,
        quot => dout,
        remd => sig_remd,
        clk => clk,
        ce => ce,
        reset => reset);

end architecture;



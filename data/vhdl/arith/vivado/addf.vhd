library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity addf is
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

architecture arch of addf is

  component array_RAM_fadd_32bkb is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 10;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 32
    );
    port (
      clk   : in  std_logic;
      reset : in  std_logic;
      ce    : in  std_logic;
      din0  : in  std_logic_vector(din0_WIDTH - 1 downto 0);
      din1  : in  std_logic_vector(din1_WIDTH - 1 downto 0);
      dout  : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid                         : std_logic;
  signal out_array                          : std_logic_vector(1 downto 0);
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic;

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

  buff : entity work.delay_buffer(arch) generic map(8)
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

  array_RAM_fadd_32ns_32ns_32_10_full_dsp_1_U1 : component array_RAM_fadd_32bkb
    port map(
      clk   => clk,
      reset => rst,
      ce    => oehb_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result
    );

end architecture;


-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.2.1 (64-bit)
-- Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
Library ieee;
use ieee.std_logic_1164.all;

entity array_RAM_fadd_32bkb is
    generic (
        ID         : integer := 1;
        NUM_STAGE  : integer := 10;
        din0_WIDTH : integer := 32;
        din1_WIDTH : integer := 32;
        dout_WIDTH : integer := 32
    );
    port (
        clk   : in  std_logic;
        reset : in  std_logic;
        ce    : in  std_logic;
        din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
        din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
        dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
    );
end entity;

architecture arch of array_RAM_fadd_32bkb is
    --------------------- Component ---------------------
    component array_RAM_ap_fadd_8_full_dsp_32 is
        port (
            aclk                 : in  std_logic;
            aclken               : in  std_logic;
            s_axis_a_tvalid      : in  std_logic;
            s_axis_a_tdata       : in  std_logic_vector(31 downto 0);
            s_axis_b_tvalid      : in  std_logic;
            s_axis_b_tdata       : in  std_logic_vector(31 downto 0);
            m_axis_result_tvalid : out std_logic;
            m_axis_result_tdata  : out std_logic_vector(31 downto 0)
        );
    end component;
    --------------------- Local signal ------------------
    signal aclk      : std_logic;
    signal aclken    : std_logic;
    signal a_tvalid  : std_logic;
    signal a_tdata   : std_logic_vector(31 downto 0);
    signal b_tvalid  : std_logic;
    signal b_tdata   : std_logic_vector(31 downto 0);
    signal r_tvalid  : std_logic;
    signal r_tdata   : std_logic_vector(31 downto 0);
    signal din0_buf1 : std_logic_vector(din0_WIDTH-1 downto 0);
    signal din1_buf1 : std_logic_vector(din1_WIDTH-1 downto 0);
    signal ce_r      : std_logic;
    signal dout_i    : std_logic_vector(dout_WIDTH-1 downto 0);
    signal dout_r    : std_logic_vector(dout_WIDTH-1 downto 0);
begin
    --------------------- Instantiation -----------------
    array_RAM_ap_fadd_8_full_dsp_32_u : component array_RAM_ap_fadd_8_full_dsp_32
    port map (
        aclk                 => aclk,
        aclken               => aclken,
        s_axis_a_tvalid      => a_tvalid,
        s_axis_a_tdata       => a_tdata,
        s_axis_b_tvalid      => b_tvalid,
        s_axis_b_tdata       => b_tdata,
        m_axis_result_tvalid => r_tvalid,
        m_axis_result_tdata  => r_tdata
    );

    --------------------- Assignment --------------------
    aclk     <= clk;
    aclken   <= ce_r;
    a_tvalid <= '1';
    a_tdata  <= din0_buf1;
    b_tvalid <= '1';
    b_tdata  <= din1_buf1;
    dout_i   <= r_tdata;

    --------------------- Input buffer ------------------
    process (clk) begin
        if clk'event and clk = '1' then
            if ce = '1' then
                din0_buf1 <= din0;
                din1_buf1 <= din1;
            end if;
        end if;
    end process;

    process (clk) begin
        if clk'event and clk = '1' then
            ce_r <= ce;
        end if;
    end process;

    process (clk) begin
        if clk'event and clk = '1' then
            if ce_r = '1' then
                dout_r <= dout_i;
            end if;
        end if;
    end process;

    dout <= dout_i when ce_r = '1' else dout_r;
end architecture;

-- (c) Copyright 1995-2019 Xilinx, Inc. All rights reserved.
-- 
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
-- 
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
-- 
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
-- 
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
-- 
-- DO NOT MODIFY THIS FILE.

-- IP VLNV: xilinx.com:ip:floating_point:7.1
-- IP Revision: 8

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

LIBRARY floating_point_v7_1_8;
USE floating_point_v7_1_8.floating_point_v7_1_8;

ENTITY array_RAM_ap_fadd_8_full_dsp_32 IS
  PORT (
    aclk : IN STD_LOGIC;
    aclken : IN STD_LOGIC;
    s_axis_a_tvalid : IN STD_LOGIC;
    s_axis_a_tdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    s_axis_b_tvalid : IN STD_LOGIC;
    s_axis_b_tdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    m_axis_result_tvalid : OUT STD_LOGIC;
    m_axis_result_tdata : OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
  );
END array_RAM_ap_fadd_8_full_dsp_32;

ARCHITECTURE array_RAM_ap_fadd_8_full_dsp_32_arch OF array_RAM_ap_fadd_8_full_dsp_32 IS
  ATTRIBUTE DowngradeIPIdentifiedWarnings : STRING;
  ATTRIBUTE DowngradeIPIdentifiedWarnings OF array_RAM_ap_fadd_8_full_dsp_32_arch: ARCHITECTURE IS "yes";
  COMPONENT floating_point_v7_1_8 IS
    GENERIC (
      C_XDEVICEFAMILY : STRING;
      C_HAS_ADD : INTEGER;
      C_HAS_SUBTRACT : INTEGER;
      C_HAS_MULTIPLY : INTEGER;
      C_HAS_DIVIDE : INTEGER;
      C_HAS_SQRT : INTEGER;
      C_HAS_COMPARE : INTEGER;
      C_HAS_FIX_TO_FLT : INTEGER;
      C_HAS_FLT_TO_FIX : INTEGER;
      C_HAS_FLT_TO_FLT : INTEGER;
      C_HAS_RECIP : INTEGER;
      C_HAS_RECIP_SQRT : INTEGER;
      C_HAS_ABSOLUTE : INTEGER;
      C_HAS_LOGARITHM : INTEGER;
      C_HAS_EXPONENTIAL : INTEGER;
      C_HAS_FMA : INTEGER;
      C_HAS_FMS : INTEGER;
      C_HAS_UNFUSED_MULTIPLY_ADD : INTEGER;
      C_HAS_UNFUSED_MULTIPLY_SUB : INTEGER;
      C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_A : INTEGER;
      C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_S : INTEGER;
      C_HAS_ACCUMULATOR_A : INTEGER;
      C_HAS_ACCUMULATOR_S : INTEGER;
      C_HAS_ACCUMULATOR_PRIMITIVE_A : INTEGER;
      C_HAS_ACCUMULATOR_PRIMITIVE_S : INTEGER;
      C_A_WIDTH : INTEGER;
      C_A_FRACTION_WIDTH : INTEGER;
      C_B_WIDTH : INTEGER;
      C_B_FRACTION_WIDTH : INTEGER;
      C_C_WIDTH : INTEGER;
      C_C_FRACTION_WIDTH : INTEGER;
      C_RESULT_WIDTH : INTEGER;
      C_RESULT_FRACTION_WIDTH : INTEGER;
      C_COMPARE_OPERATION : INTEGER;
      C_LATENCY : INTEGER;
      C_OPTIMIZATION : INTEGER;
      C_MULT_USAGE : INTEGER;
      C_BRAM_USAGE : INTEGER;
      C_RATE : INTEGER;
      C_ACCUM_INPUT_MSB : INTEGER;
      C_ACCUM_MSB : INTEGER;
      C_ACCUM_LSB : INTEGER;
      C_HAS_UNDERFLOW : INTEGER;
      C_HAS_OVERFLOW : INTEGER;
      C_HAS_INVALID_OP : INTEGER;
      C_HAS_DIVIDE_BY_ZERO : INTEGER;
      C_HAS_ACCUM_OVERFLOW : INTEGER;
      C_HAS_ACCUM_INPUT_OVERFLOW : INTEGER;
      C_HAS_ACLKEN : INTEGER;
      C_HAS_ARESETN : INTEGER;
      C_THROTTLE_SCHEME : INTEGER;
      C_HAS_A_TUSER : INTEGER;
      C_HAS_A_TLAST : INTEGER;
      C_HAS_B : INTEGER;
      C_HAS_B_TUSER : INTEGER;
      C_HAS_B_TLAST : INTEGER;
      C_HAS_C : INTEGER;
      C_HAS_C_TUSER : INTEGER;
      C_HAS_C_TLAST : INTEGER;
      C_HAS_OPERATION : INTEGER;
      C_HAS_OPERATION_TUSER : INTEGER;
      C_HAS_OPERATION_TLAST : INTEGER;
      C_HAS_RESULT_TUSER : INTEGER;
      C_HAS_RESULT_TLAST : INTEGER;
      C_TLAST_RESOLUTION : INTEGER;
      C_A_TDATA_WIDTH : INTEGER;
      C_A_TUSER_WIDTH : INTEGER;
      C_B_TDATA_WIDTH : INTEGER;
      C_B_TUSER_WIDTH : INTEGER;
      C_C_TDATA_WIDTH : INTEGER;
      C_C_TUSER_WIDTH : INTEGER;
      C_OPERATION_TDATA_WIDTH : INTEGER;
      C_OPERATION_TUSER_WIDTH : INTEGER;
      C_RESULT_TDATA_WIDTH : INTEGER;
      C_RESULT_TUSER_WIDTH : INTEGER;
      C_FIXED_DATA_UNSIGNED : INTEGER
    );
    PORT (
      aclk : IN STD_LOGIC;
      aclken : IN STD_LOGIC;
      aresetn : IN STD_LOGIC;
      s_axis_a_tvalid : IN STD_LOGIC;
      s_axis_a_tready : OUT STD_LOGIC;
      s_axis_a_tdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
      s_axis_a_tuser : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
      s_axis_a_tlast : IN STD_LOGIC;
      s_axis_b_tvalid : IN STD_LOGIC;
      s_axis_b_tready : OUT STD_LOGIC;
      s_axis_b_tdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
      s_axis_b_tuser : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
      s_axis_b_tlast : IN STD_LOGIC;
      s_axis_c_tvalid : IN STD_LOGIC;
      s_axis_c_tready : OUT STD_LOGIC;
      s_axis_c_tdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
      s_axis_c_tuser : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
      s_axis_c_tlast : IN STD_LOGIC;
      s_axis_operation_tvalid : IN STD_LOGIC;
      s_axis_operation_tready : OUT STD_LOGIC;
      s_axis_operation_tdata : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      s_axis_operation_tuser : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
      s_axis_operation_tlast : IN STD_LOGIC;
      m_axis_result_tvalid : OUT STD_LOGIC;
      m_axis_result_tready : IN STD_LOGIC;
      m_axis_result_tdata : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
      m_axis_result_tuser : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
      m_axis_result_tlast : OUT STD_LOGIC
    );
  END COMPONENT floating_point_v7_1_8;
  ATTRIBUTE X_INTERFACE_INFO : STRING;
  ATTRIBUTE X_INTERFACE_PARAMETER : STRING;
  ATTRIBUTE X_INTERFACE_INFO OF m_axis_result_tdata: SIGNAL IS "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TDATA";
  ATTRIBUTE X_INTERFACE_PARAMETER OF m_axis_result_tvalid: SIGNAL IS "XIL_INTERFACENAME M_AXIS_RESULT, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.000, LAYERED_METADATA undef, INSERT_VIP 0";
  ATTRIBUTE X_INTERFACE_INFO OF m_axis_result_tvalid: SIGNAL IS "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TVALID";
  ATTRIBUTE X_INTERFACE_INFO OF s_axis_b_tdata: SIGNAL IS "xilinx.com:interface:axis:1.0 S_AXIS_B TDATA";
  ATTRIBUTE X_INTERFACE_PARAMETER OF s_axis_b_tvalid: SIGNAL IS "XIL_INTERFACENAME S_AXIS_B, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.000, LAYERED_METADATA undef, INSERT_VIP 0";
  ATTRIBUTE X_INTERFACE_INFO OF s_axis_b_tvalid: SIGNAL IS "xilinx.com:interface:axis:1.0 S_AXIS_B TVALID";
  ATTRIBUTE X_INTERFACE_INFO OF s_axis_a_tdata: SIGNAL IS "xilinx.com:interface:axis:1.0 S_AXIS_A TDATA";
  ATTRIBUTE X_INTERFACE_PARAMETER OF s_axis_a_tvalid: SIGNAL IS "XIL_INTERFACENAME S_AXIS_A, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.000, LAYERED_METADATA undef, INSERT_VIP 0";
  ATTRIBUTE X_INTERFACE_INFO OF s_axis_a_tvalid: SIGNAL IS "xilinx.com:interface:axis:1.0 S_AXIS_A TVALID";
  ATTRIBUTE X_INTERFACE_PARAMETER OF aclken: SIGNAL IS "XIL_INTERFACENAME aclken_intf, POLARITY ACTIVE_LOW";
  ATTRIBUTE X_INTERFACE_INFO OF aclken: SIGNAL IS "xilinx.com:signal:clockenable:1.0 aclken_intf CE";
  ATTRIBUTE X_INTERFACE_PARAMETER OF aclk: SIGNAL IS "XIL_INTERFACENAME aclk_intf, ASSOCIATED_BUSIF S_AXIS_OPERATION:M_AXIS_RESULT:S_AXIS_C:S_AXIS_B:S_AXIS_A, ASSOCIATED_RESET aresetn, ASSOCIATED_CLKEN aclken, FREQ_HZ 10000000, PHASE 0.000, INSERT_VIP 0";
  ATTRIBUTE X_INTERFACE_INFO OF aclk: SIGNAL IS "xilinx.com:signal:clock:1.0 aclk_intf CLK";
BEGIN
  U0 : floating_point_v7_1_8
    GENERIC MAP (
      C_XDEVICEFAMILY => "kintex7",
      C_HAS_ADD => 1,
      C_HAS_SUBTRACT => 0,
      C_HAS_MULTIPLY => 0,
      C_HAS_DIVIDE => 0,
      C_HAS_SQRT => 0,
      C_HAS_COMPARE => 0,
      C_HAS_FIX_TO_FLT => 0,
      C_HAS_FLT_TO_FIX => 0,
      C_HAS_FLT_TO_FLT => 0,
      C_HAS_RECIP => 0,
      C_HAS_RECIP_SQRT => 0,
      C_HAS_ABSOLUTE => 0,
      C_HAS_LOGARITHM => 0,
      C_HAS_EXPONENTIAL => 0,
      C_HAS_FMA => 0,
      C_HAS_FMS => 0,
      C_HAS_UNFUSED_MULTIPLY_ADD => 0,
      C_HAS_UNFUSED_MULTIPLY_SUB => 0,
      C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_A => 0,
      C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_S => 0,
      C_HAS_ACCUMULATOR_A => 0,
      C_HAS_ACCUMULATOR_S => 0,
      C_HAS_ACCUMULATOR_PRIMITIVE_A => 0,
      C_HAS_ACCUMULATOR_PRIMITIVE_S => 0,
      C_A_WIDTH => 32,
      C_A_FRACTION_WIDTH => 24,
      C_B_WIDTH => 32,
      C_B_FRACTION_WIDTH => 24,
      C_C_WIDTH => 32,
      C_C_FRACTION_WIDTH => 24,
      C_RESULT_WIDTH => 32,
      C_RESULT_FRACTION_WIDTH => 24,
      C_COMPARE_OPERATION => 8,
      C_LATENCY => 8,
      C_OPTIMIZATION => 1,
      C_MULT_USAGE => 2,
      C_BRAM_USAGE => 0,
      C_RATE => 1,
      C_ACCUM_INPUT_MSB => 32,
      C_ACCUM_MSB => 32,
      C_ACCUM_LSB => -31,
      C_HAS_UNDERFLOW => 0,
      C_HAS_OVERFLOW => 0,
      C_HAS_INVALID_OP => 0,
      C_HAS_DIVIDE_BY_ZERO => 0,
      C_HAS_ACCUM_OVERFLOW => 0,
      C_HAS_ACCUM_INPUT_OVERFLOW => 0,
      C_HAS_ACLKEN => 1,
      C_HAS_ARESETN => 0,
      C_THROTTLE_SCHEME => 3,
      C_HAS_A_TUSER => 0,
      C_HAS_A_TLAST => 0,
      C_HAS_B => 1,
      C_HAS_B_TUSER => 0,
      C_HAS_B_TLAST => 0,
      C_HAS_C => 0,
      C_HAS_C_TUSER => 0,
      C_HAS_C_TLAST => 0,
      C_HAS_OPERATION => 0,
      C_HAS_OPERATION_TUSER => 0,
      C_HAS_OPERATION_TLAST => 0,
      C_HAS_RESULT_TUSER => 0,
      C_HAS_RESULT_TLAST => 0,
      C_TLAST_RESOLUTION => 0,
      C_A_TDATA_WIDTH => 32,
      C_A_TUSER_WIDTH => 1,
      C_B_TDATA_WIDTH => 32,
      C_B_TUSER_WIDTH => 1,
      C_C_TDATA_WIDTH => 32,
      C_C_TUSER_WIDTH => 1,
      C_OPERATION_TDATA_WIDTH => 8,
      C_OPERATION_TUSER_WIDTH => 1,
      C_RESULT_TDATA_WIDTH => 32,
      C_RESULT_TUSER_WIDTH => 1,
      C_FIXED_DATA_UNSIGNED => 0
    )
    PORT MAP (
      aclk => aclk,
      aclken => aclken,
      aresetn => '1',
      s_axis_a_tvalid => s_axis_a_tvalid,
      s_axis_a_tdata => s_axis_a_tdata,
      s_axis_a_tuser => STD_LOGIC_VECTOR(TO_UNSIGNED(0, 1)),
      s_axis_a_tlast => '0',
      s_axis_b_tvalid => s_axis_b_tvalid,
      s_axis_b_tdata => s_axis_b_tdata,
      s_axis_b_tuser => STD_LOGIC_VECTOR(TO_UNSIGNED(0, 1)),
      s_axis_b_tlast => '0',
      s_axis_c_tvalid => '0',
      s_axis_c_tdata => STD_LOGIC_VECTOR(TO_UNSIGNED(0, 32)),
      s_axis_c_tuser => STD_LOGIC_VECTOR(TO_UNSIGNED(0, 1)),
      s_axis_c_tlast => '0',
      s_axis_operation_tvalid => '0',
      s_axis_operation_tdata => STD_LOGIC_VECTOR(TO_UNSIGNED(0, 8)),
      s_axis_operation_tuser => STD_LOGIC_VECTOR(TO_UNSIGNED(0, 1)),
      s_axis_operation_tlast => '0',
      m_axis_result_tvalid => m_axis_result_tvalid,
      m_axis_result_tready => '0',
      m_axis_result_tdata => m_axis_result_tdata
    );
END array_RAM_ap_fadd_8_full_dsp_32_arch;
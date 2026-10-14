-- Brief: This file includes the IP core and the wrapper for the following
-- arithmetic operation:
-- SqrtOp (single precision)
--
-- The design was generated with Vitis HLS 2023.2 from the C++ function below
-- (part xc7k160tfbg484-2, clock period 6 ns, II = 1, clock enable):
--
--   #include <hls_math.h>
--   float sqrtf_top(float x) {
--   #pragma HLS INTERFACE mode=ap_ctrl_none port=return
--   #pragma HLS INTERFACE mode=ap_none port=x
--   #pragma HLS PIPELINE II=1
--     return hls::sqrt(x);
--   }
--
--   config_interface -clock_enable=true
--
-- Vitis HLS implements the square root with the Xilinx floating-point IP core
-- (fabric only, no DSP, latency 10). Its instance below uses the same
-- configuration, but version 7.1 rev. 8 of the core (from Vivado 2019.1),
-- which is the version precompiled for simulation and used by the other
-- floating-point units in data/vhdl/support/vivado_ip_wrappers.vhd.
--
-- Total latency: 11 cycles
-- > 1 (input register)
-- > 10 (floating-point core)

library ieee;
use ieee.std_logic_1164.all;
library floating_point_v7_1_8;
use floating_point_v7_1_8.floating_point_v7_1_8;

entity sqrtf_vitis_hls_single_precision_lat_10 is
  port (
    aclk : in std_logic;
    aclken : in std_logic;
    s_axis_a_tvalid : in std_logic;
    s_axis_a_tdata : in std_logic_vector(31 downto 0);
    m_axis_result_tvalid : out std_logic;
    m_axis_result_tdata : out std_logic_vector(31 downto 0)
  );
end entity;

architecture arch of sqrtf_vitis_hls_single_precision_lat_10 is
begin
  inst : entity floating_point_v7_1_8.floating_point_v7_1_8
    generic map (
      C_XDEVICEFAMILY => "kintex7",
      C_HAS_ADD => 0,
      C_HAS_SUBTRACT => 0,
      C_HAS_MULTIPLY => 0,
      C_HAS_DIVIDE => 0,
      C_HAS_SQRT => 1,
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
      C_LATENCY => 10,
      C_OPTIMIZATION => 1,
      C_MULT_USAGE => 0,
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
      C_HAS_B => 0,
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
    port map (
      aclk => aclk,
      aclken => aclken,
      aresetn => '1',
      s_axis_a_tvalid => s_axis_a_tvalid,
      s_axis_a_tready => open,
      s_axis_a_tdata => s_axis_a_tdata,
      s_axis_a_tuser => (others => '0'),
      s_axis_a_tlast => '0',
      s_axis_b_tvalid => '0',
      s_axis_b_tready => open,
      s_axis_b_tdata => "00000000000000000000000000000000",
      s_axis_b_tuser => (others => '0'),
      s_axis_b_tlast => '0',
      s_axis_c_tvalid => '0',
      s_axis_c_tready => open,
      s_axis_c_tdata => "00000000000000000000000000000000",
      s_axis_c_tuser => (others => '0'),
      s_axis_c_tlast => '0',
      s_axis_operation_tvalid => '0',
      s_axis_operation_tready => open,
      s_axis_operation_tdata => "00000000",
      s_axis_operation_tuser => (others => '0'),
      s_axis_operation_tlast => '0',
      m_axis_result_tvalid => m_axis_result_tvalid,
      m_axis_result_tready => '0',
      m_axis_result_tdata => m_axis_result_tdata,
      m_axis_result_tuser => open,
      m_axis_result_tlast => open
    );
end architecture;

-- This is the module that is instantiated in the handshake unit.
-- See `tools/unit-generators/vhdl/generators/handshake/sqrtf.py` for its instantiation logic
library ieee;
use ieee.std_logic_1164.all;

entity sqrtf_vitis_hls_wrapper is
  port (
    clk   : in  std_logic;
    reset : in  std_logic;
    ce    : in  std_logic;
    din0  : in  std_logic_vector(32 - 1 downto 0);
    dout  : out std_logic_vector(32 - 1 downto 0)
  );
end entity;

architecture arch of sqrtf_vitis_hls_wrapper is
  signal din0_buf1 : std_logic_vector(32 - 1 downto 0);
  signal ce_r      : std_logic;
  signal dout_i    : std_logic_vector(32 - 1 downto 0);
  signal dout_r    : std_logic_vector(32 - 1 downto 0);
begin
  sqrtf_vitis_hls_single_precision_lat_10_u : entity work.sqrtf_vitis_hls_single_precision_lat_10
    port map (
      aclk                 => clk,
      aclken               => ce_r,
      s_axis_a_tvalid      => '1',
      s_axis_a_tdata       => din0_buf1,
      m_axis_result_tvalid => open,
      m_axis_result_tdata  => dout_i
    );

  -- Input register
  process (clk) begin
    if rising_edge(clk) then
      if ce = '1' then
        din0_buf1 <= din0;
      end if;
    end if;
  end process;

  process (clk) begin
    if rising_edge(clk) then
      ce_r <= ce;
    end if;
  end process;

  -- Hold the output while the pipeline is stalled
  process (clk) begin
    if rising_edge(clk) then
      if ce_r = '1' then
        dout_r <= dout_i;
      end if;
    end if;
  end process;

  dout <= dout_i when ce_r = '1' else dout_r;
end architecture;

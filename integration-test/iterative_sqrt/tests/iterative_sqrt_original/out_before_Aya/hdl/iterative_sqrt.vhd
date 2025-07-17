library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity iterative_sqrt is
  port (
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    n_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of iterative_sqrt is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(31 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(31 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi0_outs : std_logic_vector(31 downto 0);
  signal extsi0_outs_valid : std_logic;
  signal extsi0_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(0 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal cond_br35_trueOut : std_logic_vector(31 downto 0);
  signal cond_br35_trueOut_valid : std_logic;
  signal cond_br35_trueOut_ready : std_logic;
  signal cond_br35_falseOut : std_logic_vector(31 downto 0);
  signal cond_br35_falseOut_valid : std_logic;
  signal cond_br35_falseOut_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal cond_br36_trueOut : std_logic_vector(31 downto 0);
  signal cond_br36_trueOut_valid : std_logic;
  signal cond_br36_trueOut_ready : std_logic;
  signal cond_br36_falseOut : std_logic_vector(31 downto 0);
  signal cond_br36_falseOut_valid : std_logic;
  signal cond_br36_falseOut_ready : std_logic;
  signal constant1_outs : std_logic_vector(0 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal merge0_outs : std_logic_vector(0 downto 0);
  signal merge0_outs_valid : std_logic;
  signal merge0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(0 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(0 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(0 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal fork4_outs_3 : std_logic_vector(0 downto 0);
  signal fork4_outs_3_valid : std_logic;
  signal fork4_outs_3_ready : std_logic;
  signal fork4_outs_4 : std_logic_vector(0 downto 0);
  signal fork4_outs_4_valid : std_logic;
  signal fork4_outs_4_ready : std_logic;
  signal fork4_outs_5 : std_logic_vector(0 downto 0);
  signal fork4_outs_5_valid : std_logic;
  signal fork4_outs_5_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(0 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(31 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(0 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal mux10_outs : std_logic_vector(31 downto 0);
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_2 : std_logic_vector(31 downto 0);
  signal fork5_outs_2_valid : std_logic;
  signal fork5_outs_2_ready : std_logic;
  signal fork5_outs_3 : std_logic_vector(31 downto 0);
  signal fork5_outs_3_valid : std_logic;
  signal fork5_outs_3_ready : std_logic;
  signal mux11_outs : std_logic_vector(31 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(31 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal fork6_outs_3 : std_logic_vector(31 downto 0);
  signal fork6_outs_3_valid : std_logic;
  signal fork6_outs_3_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(0 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal fork7_outs_3 : std_logic_vector(0 downto 0);
  signal fork7_outs_3_valid : std_logic;
  signal fork7_outs_3_ready : std_logic;
  signal fork7_outs_4 : std_logic_vector(0 downto 0);
  signal fork7_outs_4_valid : std_logic;
  signal fork7_outs_4_ready : std_logic;
  signal fork7_outs_5 : std_logic_vector(0 downto 0);
  signal fork7_outs_5_valid : std_logic;
  signal fork7_outs_5_ready : std_logic;
  signal fork7_outs_6 : std_logic_vector(0 downto 0);
  signal fork7_outs_6_valid : std_logic;
  signal fork7_outs_6_ready : std_logic;
  signal fork7_outs_7 : std_logic_vector(0 downto 0);
  signal fork7_outs_7_valid : std_logic;
  signal fork7_outs_7_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal cond_br37_trueOut : std_logic_vector(31 downto 0);
  signal cond_br37_trueOut_valid : std_logic;
  signal cond_br37_trueOut_ready : std_logic;
  signal cond_br37_falseOut : std_logic_vector(31 downto 0);
  signal cond_br37_falseOut_valid : std_logic;
  signal cond_br37_falseOut_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(0 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal cond_br38_trueOut : std_logic_vector(0 downto 0);
  signal cond_br38_trueOut_valid : std_logic;
  signal cond_br38_trueOut_ready : std_logic;
  signal cond_br38_falseOut : std_logic_vector(0 downto 0);
  signal cond_br38_falseOut_valid : std_logic;
  signal cond_br38_falseOut_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br39_trueOut : std_logic_vector(31 downto 0);
  signal cond_br39_trueOut_valid : std_logic;
  signal cond_br39_trueOut_ready : std_logic;
  signal cond_br39_falseOut : std_logic_vector(31 downto 0);
  signal cond_br39_falseOut_valid : std_logic;
  signal cond_br39_falseOut_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(31 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork8_outs_3 : std_logic_vector(31 downto 0);
  signal fork8_outs_3_valid : std_logic;
  signal fork8_outs_3_ready : std_logic;
  signal cond_br40_trueOut : std_logic_vector(31 downto 0);
  signal cond_br40_trueOut_valid : std_logic;
  signal cond_br40_trueOut_ready : std_logic;
  signal cond_br40_falseOut : std_logic_vector(31 downto 0);
  signal cond_br40_falseOut_valid : std_logic;
  signal cond_br40_falseOut_ready : std_logic;
  signal cond_br41_trueOut : std_logic_vector(31 downto 0);
  signal cond_br41_trueOut_valid : std_logic;
  signal cond_br41_trueOut_ready : std_logic;
  signal cond_br41_falseOut : std_logic_vector(31 downto 0);
  signal cond_br41_falseOut_valid : std_logic;
  signal cond_br41_falseOut_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(1 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal shrsi0_result : std_logic_vector(31 downto 0);
  signal shrsi0_result_valid : std_logic;
  signal shrsi0_result_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork9_outs_2 : std_logic_vector(31 downto 0);
  signal fork9_outs_2_valid : std_logic;
  signal fork9_outs_2_ready : std_logic;
  signal fork9_outs_3 : std_logic_vector(31 downto 0);
  signal fork9_outs_3_valid : std_logic;
  signal fork9_outs_3_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal fork10_outs_2 : std_logic_vector(31 downto 0);
  signal fork10_outs_2_valid : std_logic;
  signal fork10_outs_2_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(0 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal fork11_outs_3 : std_logic_vector(0 downto 0);
  signal fork11_outs_3_valid : std_logic;
  signal fork11_outs_3_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal select0_result : std_logic_vector(31 downto 0);
  signal select0_result_valid : std_logic;
  signal select0_result_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(0 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(0 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal fork12_outs_2 : std_logic_vector(0 downto 0);
  signal fork12_outs_2_valid : std_logic;
  signal fork12_outs_2_ready : std_logic;
  signal cond_br42_trueOut : std_logic_vector(31 downto 0);
  signal cond_br42_trueOut_valid : std_logic;
  signal cond_br42_trueOut_ready : std_logic;
  signal cond_br42_falseOut : std_logic_vector(31 downto 0);
  signal cond_br42_falseOut_valid : std_logic;
  signal cond_br42_falseOut_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(1 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(31 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(31 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal select1_result : std_logic_vector(31 downto 0);
  signal select1_result_valid : std_logic;
  signal select1_result_ready : std_logic;

begin

  out0 <= select1_result;
  out0_valid <= select1_result_valid;
  select1_result_ready <= out0_ready;
  end_valid <= fork0_outs_0_valid;
  fork0_outs_0_ready <= end_ready;

  fork0 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => n,
      ins_valid => n_valid,
      ins_ready => n_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant2 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi0 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi0_outs,
      outs_valid => extsi0_outs_valid,
      outs_ready => extsi0_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant3 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  buffer8 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  buffer9 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  cond_br35 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_5,
      condition_valid => fork7_outs_5_valid,
      condition_ready => fork7_outs_5_ready,
      data => buffer9_outs,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br35_trueOut,
      trueOut_valid => cond_br35_trueOut_valid,
      trueOut_ready => cond_br35_trueOut_ready,
      falseOut => cond_br35_falseOut,
      falseOut_valid => cond_br35_falseOut_valid,
      falseOut_ready => cond_br35_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br35_falseOut,
      ins_valid => cond_br35_falseOut_valid,
      ins_ready => cond_br35_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer4 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer5 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  cond_br36 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_6,
      condition_valid => fork7_outs_6_valid,
      condition_ready => fork7_outs_6_ready,
      data => buffer5_outs,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br36_trueOut,
      trueOut_valid => cond_br36_trueOut_valid,
      trueOut_ready => cond_br36_trueOut_ready,
      falseOut => cond_br36_falseOut,
      falseOut_valid => cond_br36_falseOut_valid,
      falseOut_ready => cond_br36_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br36_falseOut,
      ins_valid => cond_br36_falseOut_valid,
      ins_ready => cond_br36_falseOut_ready,
      clk => clk,
      rst => rst
    );

  constant1 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  extsi1 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  merge0 : entity work.merge(arch) generic map(2, 1)
    port map(
      ins(0) => fork3_outs_0,
      ins(1) => fork7_outs_7,
      ins_valid(0) => fork3_outs_0_valid,
      ins_valid(1) => fork7_outs_7_valid,
      ins_ready(0) => fork3_outs_0_ready,
      ins_ready(1) => fork7_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => merge0_outs,
      outs_valid => merge0_outs_valid,
      outs_ready => merge0_outs_ready
    );

  buffer0 : entity work.oehb(arch) generic map(1)
    port map(
      ins => merge0_outs,
      ins_valid => merge0_outs_valid,
      ins_ready => merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  buffer1 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs(2) => fork4_outs_2,
      outs(3) => fork4_outs_3,
      outs(4) => fork4_outs_4,
      outs(5) => fork4_outs_5,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_valid(2) => fork4_outs_2_valid,
      outs_valid(3) => fork4_outs_3_valid,
      outs_valid(4) => fork4_outs_4_valid,
      outs_valid(5) => fork4_outs_5_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_ready(2) => fork4_outs_2_ready,
      outs_ready(3) => fork4_outs_3_ready,
      outs_ready(4) => fork4_outs_4_ready,
      outs_ready(5) => fork4_outs_5_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_5,
      index_valid => fork4_outs_5_valid,
      index_ready => fork4_outs_5_ready,
      ins(0) => fork1_outs_1,
      ins(1) => fork8_outs_3,
      ins_valid(0) => fork1_outs_1_valid,
      ins_valid(1) => fork8_outs_3_valid,
      ins_ready(0) => fork1_outs_1_ready,
      ins_ready(1) => fork8_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer6 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  buffer7 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_1,
      index_valid => fork11_outs_1_valid,
      index_ready => fork11_outs_1_ready,
      ins(0) => fork5_outs_2,
      ins(1) => buffer7_outs,
      ins_valid(0) => fork5_outs_2_valid,
      ins_valid(1) => buffer7_outs_valid,
      ins_ready(0) => fork5_outs_2_ready,
      ins_ready(1) => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork12_outs_1,
      index_valid => fork12_outs_1_valid,
      index_ready => fork12_outs_1_ready,
      ins(0) => fork5_outs_3,
      ins(1) => addi1_result,
      ins_valid(0) => fork5_outs_3_valid,
      ins_valid(1) => addi1_result_valid,
      ins_ready(0) => fork5_outs_3_ready,
      ins_ready(1) => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  buffer10 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer11 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_2,
      index_valid => fork11_outs_2_valid,
      index_ready => fork11_outs_2_ready,
      ins(0) => fork6_outs_2,
      ins(1) => buffer11_outs,
      ins_valid(0) => fork6_outs_2_valid,
      ins_valid(1) => buffer11_outs_valid,
      ins_ready(0) => fork6_outs_2_ready,
      ins_ready(1) => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork12_outs_2,
      index_valid => fork12_outs_2_valid,
      index_ready => fork12_outs_2_ready,
      ins(0) => addi2_result,
      ins(1) => fork6_outs_3,
      ins_valid(0) => addi2_result_valid,
      ins_valid(1) => fork6_outs_3_valid,
      ins_ready(0) => addi2_result_ready,
      ins_ready(1) => fork6_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork4_outs_4,
      index_valid => fork4_outs_4_valid,
      index_ready => fork4_outs_4_ready,
      ins(0) => fork2_outs_1,
      ins(1) => fork11_outs_3,
      ins_valid(0) => fork2_outs_1_valid,
      ins_valid(1) => fork11_outs_3_valid,
      ins_ready(0) => fork2_outs_1_ready,
      ins_ready(1) => fork11_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_3,
      index_valid => fork4_outs_3_valid,
      index_ready => fork4_outs_3_ready,
      ins(0) => extsi1_outs,
      ins(1) => select0_result,
      ins_valid(0) => extsi1_outs_valid,
      ins_valid(1) => select0_result_valid,
      ins_ready(0) => extsi1_outs_ready,
      ins_ready(1) => select0_result_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork4_outs_2,
      index_valid => fork4_outs_2_valid,
      index_ready => fork4_outs_2_ready,
      ins(0) => fork2_outs_0,
      ins(1) => andi1_result,
      ins_valid(0) => fork2_outs_0_valid,
      ins_valid(1) => andi1_result_valid,
      ins_ready(0) => fork2_outs_0_ready,
      ins_ready(1) => andi1_result_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  mux10 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => extsi0_outs,
      ins(1) => cond_br36_trueOut,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => cond_br36_trueOut_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => cond_br36_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux10_outs,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  buffer18 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux10_outs,
      ins_valid => mux10_outs_valid,
      ins_ready => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  buffer19 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs(2) => fork5_outs_2,
      outs(3) => fork5_outs_3,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_valid(2) => fork5_outs_2_valid,
      outs_valid(3) => fork5_outs_3_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_ready(2) => fork5_outs_2_ready,
      outs_ready(3) => fork5_outs_3_ready
    );

  mux11 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => fork1_outs_0,
      ins(1) => cond_br35_trueOut,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => cond_br35_trueOut_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => cond_br35_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  buffer20 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux11_outs,
      ins_valid => mux11_outs_valid,
      ins_ready => mux11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  buffer21 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer20_outs,
      ins_valid => buffer20_outs_valid,
      ins_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs(3) => fork6_outs_3,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_valid(3) => fork6_outs_3_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready,
      outs_ready(3) => fork6_outs_3_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork5_outs_1,
      lhs_valid => fork5_outs_1_valid,
      lhs_ready => fork5_outs_1_ready,
      rhs => fork6_outs_1,
      rhs_valid => fork6_outs_1_valid,
      rhs_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer12 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  andi0 : entity work.andi(arch) generic map(1)
    port map(
      lhs => cmpi0_result,
      lhs_valid => cmpi0_result_valid,
      lhs_ready => cmpi0_result_ready,
      rhs => buffer13_outs,
      rhs_valid => buffer13_outs_valid,
      rhs_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(8, 1)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs(3) => fork7_outs_3,
      outs(4) => fork7_outs_4,
      outs(5) => fork7_outs_5,
      outs(6) => fork7_outs_6,
      outs(7) => fork7_outs_7,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_valid(3) => fork7_outs_3_valid,
      outs_valid(4) => fork7_outs_4_valid,
      outs_valid(5) => fork7_outs_5_valid,
      outs_valid(6) => fork7_outs_6_valid,
      outs_valid(7) => fork7_outs_7_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready,
      outs_ready(3) => fork7_outs_3_ready,
      outs_ready(4) => fork7_outs_4_ready,
      outs_ready(5) => fork7_outs_5_ready,
      outs_ready(6) => fork7_outs_6_ready,
      outs_ready(7) => fork7_outs_7_ready
    );

  buffer14 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  buffer15 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  cond_br37 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_4,
      condition_valid => fork7_outs_4_valid,
      condition_ready => fork7_outs_4_ready,
      data => buffer15_outs,
      data_valid => buffer15_outs_valid,
      data_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br37_trueOut,
      trueOut_valid => cond_br37_trueOut_valid,
      trueOut_ready => cond_br37_trueOut_ready,
      falseOut => cond_br37_falseOut,
      falseOut_valid => cond_br37_falseOut_valid,
      falseOut_ready => cond_br37_falseOut_ready
    );

  buffer16 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer17 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  cond_br38 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork7_outs_3,
      condition_valid => fork7_outs_3_valid,
      condition_ready => fork7_outs_3_ready,
      data => buffer17_outs,
      data_valid => buffer17_outs_valid,
      data_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br38_trueOut,
      trueOut_valid => cond_br38_trueOut_valid,
      trueOut_ready => cond_br38_trueOut_ready,
      falseOut => cond_br38_falseOut,
      falseOut_valid => cond_br38_falseOut_valid,
      falseOut_ready => cond_br38_falseOut_ready
    );

  buffer2 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  buffer3 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  cond_br39 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_2,
      condition_valid => fork7_outs_2_valid,
      condition_ready => fork7_outs_2_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br39_trueOut,
      trueOut_valid => cond_br39_trueOut_valid,
      trueOut_ready => cond_br39_trueOut_ready,
      falseOut => cond_br39_falseOut,
      falseOut_valid => cond_br39_falseOut_valid,
      falseOut_ready => cond_br39_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br39_falseOut,
      ins_valid => cond_br39_falseOut_valid,
      ins_ready => cond_br39_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork8 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => cond_br39_trueOut,
      ins_valid => cond_br39_trueOut_valid,
      ins_ready => cond_br39_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs(3) => fork8_outs_3,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_valid(3) => fork8_outs_3_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready,
      outs_ready(3) => fork8_outs_3_ready
    );

  cond_br40 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_1,
      condition_valid => fork7_outs_1_valid,
      condition_ready => fork7_outs_1_ready,
      data => fork6_outs_0,
      data_valid => fork6_outs_0_valid,
      data_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br40_trueOut,
      trueOut_valid => cond_br40_trueOut_valid,
      trueOut_ready => cond_br40_trueOut_ready,
      falseOut => cond_br40_falseOut,
      falseOut_valid => cond_br40_falseOut_valid,
      falseOut_ready => cond_br40_falseOut_ready
    );

  cond_br41 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_0,
      condition_valid => fork7_outs_0_valid,
      condition_ready => fork7_outs_0_ready,
      data => fork5_outs_0,
      data_valid => fork5_outs_0_valid,
      data_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br41_trueOut,
      trueOut_valid => cond_br41_trueOut_valid,
      trueOut_ready => cond_br41_trueOut_ready,
      falseOut => cond_br41_falseOut,
      falseOut_valid => cond_br41_falseOut_valid,
      falseOut_ready => cond_br41_falseOut_ready
    );

  sink3 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br41_falseOut,
      ins_valid => cond_br41_falseOut_valid,
      ins_ready => cond_br41_falseOut_ready,
      clk => clk,
      rst => rst
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant7 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  extsi2 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant7_outs,
      ins_valid => constant7_outs_valid,
      ins_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => cond_br41_trueOut,
      lhs_valid => cond_br41_trueOut_valid,
      lhs_ready => cond_br41_trueOut_ready,
      rhs => cond_br40_trueOut,
      rhs_valid => cond_br40_trueOut_valid,
      rhs_ready => cond_br40_trueOut_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  shrsi0 : entity work.shrsi(arch) generic map(32)
    port map(
      lhs => addi0_result,
      lhs_valid => addi0_result_valid,
      lhs_ready => addi0_result_ready,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      clk => clk,
      rst => rst,
      result => shrsi0_result,
      result_valid => shrsi0_result_valid,
      result_ready => shrsi0_result_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => shrsi0_result,
      ins_valid => shrsi0_result_valid,
      ins_ready => shrsi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs(2) => fork9_outs_2,
      outs(3) => fork9_outs_3,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_valid(2) => fork9_outs_2_valid,
      outs_valid(3) => fork9_outs_3_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready,
      outs_ready(2) => fork9_outs_2_ready,
      outs_ready(3) => fork9_outs_3_ready
    );

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => fork9_outs_2,
      lhs_valid => fork9_outs_2_valid,
      lhs_ready => fork9_outs_2_ready,
      rhs => fork9_outs_3,
      rhs_valid => fork9_outs_3_valid,
      rhs_ready => fork9_outs_3_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => muli0_result,
      ins_valid => muli0_result_valid,
      ins_ready => muli0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs(2) => fork10_outs_2,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_valid(2) => fork10_outs_2_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready,
      outs_ready(2) => fork10_outs_2_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_2,
      lhs_valid => fork10_outs_2_valid,
      lhs_ready => fork10_outs_2_ready,
      rhs => fork8_outs_2,
      rhs_valid => fork8_outs_2_valid,
      rhs_ready => fork8_outs_2_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  cmpi2 : entity work.handshake_cmpi_2(arch) generic map(32)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => fork8_outs_1,
      rhs_valid => fork8_outs_1_valid,
      rhs_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs(2) => fork11_outs_2,
      outs(3) => fork11_outs_3,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_valid(2) => fork11_outs_2_valid,
      outs_valid(3) => fork11_outs_3_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_ready(2) => fork11_outs_2_ready,
      outs_ready(3) => fork11_outs_3_ready
    );

  andi1 : entity work.andi(arch) generic map(1)
    port map(
      lhs => fork11_outs_0,
      lhs_valid => fork11_outs_0_valid,
      lhs_ready => fork11_outs_0_ready,
      rhs => cond_br38_trueOut,
      rhs_valid => cond_br38_trueOut_valid,
      rhs_ready => cond_br38_trueOut_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  select0 : entity work.selector(arch) generic map(32)
    port map(
      condition => cmpi1_result,
      condition_valid => cmpi1_result_valid,
      condition_ready => cmpi1_result_ready,
      trueValue => fork9_outs_1,
      trueValue_valid => fork9_outs_1_valid,
      trueValue_ready => fork9_outs_1_ready,
      falseValue => cond_br37_trueOut,
      falseValue_valid => cond_br37_trueOut_valid,
      falseValue_ready => cond_br37_trueOut_ready,
      clk => clk,
      rst => rst,
      result => select0_result,
      result_valid => select0_result_valid,
      result_ready => select0_result_ready
    );

  cmpi3 : entity work.handshake_cmpi_3(arch) generic map(32)
    port map(
      lhs => fork10_outs_0,
      lhs_valid => fork10_outs_0_valid,
      lhs_ready => fork10_outs_0_ready,
      rhs => fork8_outs_0,
      rhs_valid => fork8_outs_0_valid,
      rhs_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs(2) => fork12_outs_2,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_valid(2) => fork12_outs_2_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready,
      outs_ready(2) => fork12_outs_2_ready
    );

  cond_br42 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_0,
      condition_valid => fork12_outs_0_valid,
      condition_ready => fork12_outs_0_ready,
      data => fork9_outs_0,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br42_trueOut,
      trueOut_valid => cond_br42_trueOut_valid,
      trueOut_ready => cond_br42_trueOut_ready,
      falseOut => cond_br42_falseOut,
      falseOut_valid => cond_br42_falseOut_valid,
      falseOut_ready => cond_br42_falseOut_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant8 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  extsi3 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant8_outs,
      ins_valid => constant8_outs_valid,
      ins_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => cond_br42_trueOut,
      lhs_valid => cond_br42_trueOut_valid,
      lhs_ready => cond_br42_trueOut_ready,
      rhs => extsi3_outs,
      rhs_valid => extsi3_outs_valid,
      rhs_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant6 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(32)
    port map(
      lhs => cond_br42_falseOut,
      lhs_valid => cond_br42_falseOut_valid,
      lhs_ready => cond_br42_falseOut_ready,
      rhs => constant6_outs,
      rhs_valid => constant6_outs_valid,
      rhs_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  select1 : entity work.selector(arch) generic map(32)
    port map(
      condition => cond_br38_falseOut,
      condition_valid => cond_br38_falseOut_valid,
      condition_ready => cond_br38_falseOut_ready,
      trueValue => cond_br40_falseOut,
      trueValue_valid => cond_br40_falseOut_valid,
      trueValue_ready => cond_br40_falseOut_ready,
      falseValue => cond_br37_falseOut,
      falseValue_valid => cond_br37_falseOut_valid,
      falseValue_ready => cond_br37_falseOut_ready,
      clk => clk,
      rst => rst,
      result => select1_result,
      result_valid => select1_result_valid,
      result_ready => select1_result_ready
    );

end architecture;

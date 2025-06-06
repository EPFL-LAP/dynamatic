library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity shift_reg_break_dv_dataless is
  generic(
    NUM_SLOTS : integer
  );
  port(
    -- inputs
    clk, rst   : in std_logic;
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of shift_reg_break_dv_dataless is

  signal regEn      : std_logic;
  type REG_VALID is array (0 to NUM_SLOTS - 1) of std_logic;
  signal valid_reg  : REG_VALID;

begin
  -- See 'docs/Specs/Buffering/Buffering.md'
  -- All the slots share a single handshake control and thus 
  -- accept or stall inputs together.
  process(clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        for i in 0 to NUM_SLOTS - 1 loop
          valid_reg(i) <= '0';
        end loop;
      else
        if (regEn) then
          for i in 1 to NUM_SLOTS - 1 loop
            valid_reg(i) <= valid_reg(i - 1);
          end loop;
          valid_reg(0) <= ins_valid;
        end if;               
      end if;
    end if;
  end process; 

  outs_valid <= valid_reg(NUM_SLOTS - 1);
  regEn <= not outs_valid or outs_ready;
  ins_ready <= regEn;

end architecture;
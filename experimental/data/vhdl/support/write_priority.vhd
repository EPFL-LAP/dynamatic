library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_priority is
  generic (
    ARBITER_SIZE : natural
  );
  port (
    req          : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    data_ready   : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );
end entity;

architecture arch of write_priority is

begin

  process (data_ready, req)
    variable prio_req : std_logic;

  begin
    -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
    priority_out(0) <= req(0) and data_ready(0);
    prio_req := req(0) and data_ready(0);

    for I in 1 to ARBITER_SIZE - 1 loop
      priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
      prio_req := prio_req or (req(I) and data_ready(I));
    end loop;
  end process;
end architecture;

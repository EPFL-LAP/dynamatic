library ieee;
use ieee.std_logic_1164.all;

entity and_n is
  generic (
    SIZE : integer
  );
  port (
    -- inputs
    ins : in std_logic_vector(SIZE - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

architecture arch of and_n is
  signal all_ones : std_logic_vector(SIZE - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

entity nand_n is
  generic (
    SIZE : integer
  );
  port (
    -- inputs
    ins : in std_logic_vector(SIZE - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

architecture arch of nand_n is
  signal all_ones : std_logic_vector(SIZE - 1 downto 0) := (others => '1');
begin
  outs <= '0' when ins = all_ones else '1';
end arch;

library ieee;
use ieee.std_logic_1164.all;

entity or_n is
  generic (
    SIZE : integer
  );
  port (
    -- inputs
    ins : in std_logic_vector(SIZE - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

architecture arch of or_n is
  signal all_zeros : std_logic_vector(SIZE - 1 downto 0) := (others => '0');
begin
  outs <= '0' when ins = all_zeros else '1';
end arch;

library ieee;
use ieee.std_logic_1164.all;

entity nor_n is
  generic (
    SIZE : integer
  );
  port (
    -- inputs
    ins : in std_logic_vector(SIZE - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

architecture arch of nor_n is
  signal all_zeros : std_logic_vector(SIZE - 1 downto 0) := (others => '0');
begin
  outs <= '1' when ins = all_zeros else '0';
end arch;

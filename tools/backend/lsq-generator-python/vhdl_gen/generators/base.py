from abc import ABC, abstractmethod
from vhdl_gen.configs import Configs
from vhdl_gen.context import VHDLContext


class BaseVHDLGenerator(ABC):
    """
    Abstract base class for VHDL component generators.
    It defines a common interface for all specific VHDL generator classes.
    """

    def __init__(
        self,
        ctx: VHDLContext,
        path_rtl: str,
        name: str,
        suffix: str
    ):
        """
        Initialize with a component name and optional output directory.
        """

        self.ctx = ctx
        self.path_rtl = path_rtl
        self.name = name
        self.suffix = suffix

    @abstractmethod
    def generate(self) -> None:
        """
        Generate VHDL entity and architecture, writing to file.
        """

        pass

    @abstractmethod
    def instantiate(self, **kwargs) -> str:
        """
        Abstract method to generate the VHDL component instantiation code.
        Subclasses must implement this method to return the VHDL instantiation
        code as a string, using the provided port map.
        """

        pass

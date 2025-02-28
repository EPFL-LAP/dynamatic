import re


def parse_extra_signals(extra_signals: str) -> dict[str, int]:
    """
    Parses a string of extra signals and their bitwidths.
    e.g., extra_signals = "spec: i1, tag: i8"
    """
    type_pattern = r"u?i(\d+)"
    extra_signals_dict = {}
    for signal in extra_signals.split(","):
        name, signal_type = signal.split(":")

        # Remove whitespace
        name = name.strip()
        signal_type = signal_type.strip()

        # Extract bitwidth from signal type
        match = re.match(type_pattern, signal_type)
        if match:
            bitwidth = int(match.group(1))
            extra_signals_dict[name] = bitwidth
        else:
            raise ValueError(f"Type {signal_type} of {name} is invalid")

    return extra_signals_dict


class VhdlScalarType:

    mlir_type: str
    # Note: VHDL only requires information on bitwidth and extra signals
    bitwidth: int
    extra_signals: dict[str, int]  # key: name, value: bitwidth (todo)

    def __init__(self, mlir_type: str):
        """
        Constructor for VhdlScalarType.
        Parses an incoming MLIR type string.
        """
        self.mlir_type = mlir_type

        control_pattern = r"^!handshake\.control<(?:\[([^\]]*)\])?>$"
        channel_pattern = r"^!handshake\.channel<u?i(\d+)(?:, \[([^\]]*)\])?>$"

        match = re.match(control_pattern, mlir_type)
        if match:
            self.bitwidth = 0
            if match.group(1):
                self.extra_signals = parse_extra_signals(match.group(1))
            else:
                self.extra_signals = {}
            return

        match = re.match(channel_pattern, mlir_type)
        if match:
            self.bitwidth = int(match.group(1))
            if match.group(2):
                self.extra_signals = parse_extra_signals(match.group(2))
            else:
                self.extra_signals = {}
            return

        raise ValueError(f"Type {mlir_type} is invalid")

    def has_extra_signals(self):
        return bool(self.extra_signals)

    def is_channel(self):
        return self.bitwidth > 0

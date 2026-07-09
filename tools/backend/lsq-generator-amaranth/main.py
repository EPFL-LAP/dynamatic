import json

from amaranth.back import verilog

from config import LsqConfig
from hw.lsq.group_allocator import GroupAllocator


if __name__ == "__main__":
    with open("matching_lsq1.json", "r") as f:
        config_data = json.load(f)
        config = LsqConfig.from_json(config_data)

    ga = GroupAllocator(config=config)

    with open("group_allocator.v", "w") as f:
        f.write(verilog.convert(ga, name=f"{config.name}_group_allocator"))

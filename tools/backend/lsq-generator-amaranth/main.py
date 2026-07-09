from config import LsqConfig
from group_allocator import GroupAllocator
from amaranth.back import verilog


if __name__ == "__main__":
    config = LsqConfig(
        name="lsq1",
        dataW=16,
        addrW=13,
        idW=2,
        numLdqEntries=3,
        numStqEntries=10,
        numLdPorts=3,
        numStPorts=3,
        numGroups=2,
        numLdMem=1,
        numStMem=1,
        gaNumLoads=[2, 1],
        gaNumStores=[2, 1],
        gaLdOrder=[[2, 2], [0]],
        gaLdPortIdx=[[0, 1], [2]],
        gaStPortIdx=[[0, 1], [2]],
        ldqAddrW=2,
        stqAddrW=4,
        ldpAddrW=2,
        stpAddrW=2,
        pipe0=False,
        pipe1=False,
        pipeComp=False,
        headLag=False,
        stResp=False,
        gaMulti=False,
        bypass=True
    )

    ga = GroupAllocator(config=config)

    with open("group_allocator.v", "w") as f:
        f.write(verilog.convert(ga, name=f"{config.name}_group_allocator"))

import random

MaxAddr      = 2**16
MaxValue     = 2**16
NumAddr      = 5
MaxLatInit   = 2
MaxLatPort   = 3
NumGroupInit = 1000
NumMem       = 6000
path         = './build/VHDL_SRC/'

AddrList = []
MemValue = []

def gen_g1p2():
    for i in range(0, NumAddr):
        AddrList.append(random.randint(0, MaxAddr-1))
        MemValue.append(0)

    file_init = open(path + 'init_latency.txt', 'w')
    file_sta  = open(path + 'port_sta.txt', 'w')
    file_std  = open(path + 'port_std.txt', 'w')
    file_lda  = open(path + 'port_lda.txt', 'w')
    file_ldd  = open(path + 'port_ldd.txt', 'w')
    file_ram  = open(path + 'ram_latency.txt', 'w')

    for i in range(0, NumGroupInit + 1):
        InitLat  = random.randint(0, MaxLatInit)
        file_init.write(f'{InitLat}\n')
        # store
        StaLat   = random.randint(0, MaxLatPort)
        StdLat   = random.randint(0, MaxLatPort)

        StAddrId = random.randint(0, NumAddr-1)
        StAddr   = AddrList[StAddrId]
        StData   = random.randint(0, MaxValue-1)
        MemValue[StAddrId] = StData

        file_sta.write(f'{StaLat} {StAddr}\n')
        file_std.write(f'{StdLat} {StData}\n')

        # load
        LdaLat   = random.randint(0, MaxLatPort)
        LddLat   = random.randint(0, MaxLatPort)

        LdAddrId = random.randint(0, NumAddr-1)
        LdAddr   = AddrList[LdAddrId]
        LdData   = MemValue[LdAddrId]

        file_lda.write(f'{LdaLat} {LdAddr}\n')
        file_ldd.write(f'{LddLat} {LdData}\n')

    for i in range(0, NumMem + 1):
        RamLat = random.randint(0, 1)
        file_ram.write(f'{RamLat}\n')

    file_init.close()
    file_sta.close()
    file_std.close()
    file_lda.close()
    file_ldd.close()

def gen_g2p6():
    for i in range(0, NumAddr):
        AddrList.append(random.randint(0, MaxAddr-1))
        MemValue.append(0)

    file_init  = open(path + 'init_latency.txt', 'w')
    file_sta_0 = open(path + 'port_sta_0.txt', 'w')
    file_std_0 = open(path + 'port_std_0.txt', 'w')
    file_sta_1 = open(path + 'port_sta_1.txt', 'w')
    file_std_1 = open(path + 'port_std_1.txt', 'w')
    file_sta_2 = open(path + 'port_sta_2.txt', 'w')
    file_std_2 = open(path + 'port_std_2.txt', 'w')
    file_lda_0 = open(path + 'port_lda_0.txt', 'w')
    file_ldd_0 = open(path + 'port_ldd_0.txt', 'w')
    file_lda_1 = open(path + 'port_lda_1.txt', 'w')
    file_ldd_1 = open(path + 'port_ldd_1.txt', 'w')
    file_lda_2 = open(path + 'port_lda_2.txt', 'w')
    file_ldd_2 = open(path + 'port_ldd_2.txt', 'w')
    file_ram   = open(path + 'ram_latency.txt', 'w')

    for i in range(0, NumGroupInit + 1):
        GroupId  = random.randint(0, 1)
        InitLat  = random.randint(0, MaxLatInit)
        file_init.write(f'{GroupId} {InitLat}\n')

        if (GroupId == 0):
            # store 0
            StaLat   = random.randint(0, MaxLatPort)
            StdLat   = random.randint(0, MaxLatPort)

            StAddrId = random.randint(0, NumAddr-1)
            StAddr   = AddrList[StAddrId]
            StData   = random.randint(0, MaxValue-1)
            MemValue[StAddrId] = StData

            file_sta_0.write(f'{StaLat} {StAddr}\n')
            file_std_0.write(f'{StdLat} {StData}\n')

            # load 0
            LdaLat   = random.randint(0, MaxLatPort)
            LddLat   = random.randint(0, MaxLatPort)

            LdAddrId = random.randint(0, NumAddr-1)
            LdAddr   = AddrList[LdAddrId]
            LdData   = MemValue[LdAddrId]

            file_lda_0.write(f'{LdaLat} {LdAddr}\n')
            file_ldd_0.write(f'{LddLat} {LdData}\n')

            # store 1
            StaLat   = random.randint(0, MaxLatPort)
            StdLat   = random.randint(0, MaxLatPort)

            StAddrId = random.randint(0, NumAddr-1)
            StAddr   = AddrList[StAddrId]
            StData   = random.randint(0, MaxValue-1)
            MemValue[StAddrId] = StData

            file_sta_1.write(f'{StaLat} {StAddr}\n')
            file_std_1.write(f'{StdLat} {StData}\n')

        else:
            # load 1
            LdaLat   = random.randint(0, MaxLatPort)
            LddLat   = random.randint(0, MaxLatPort)

            LdAddrId = random.randint(0, NumAddr-1)
            LdAddr   = AddrList[LdAddrId]
            LdData   = MemValue[LdAddrId]

            file_lda_1.write(f'{LdaLat} {LdAddr}\n')
            file_ldd_1.write(f'{LddLat} {LdData}\n')

            # store 2
            StaLat   = random.randint(0, MaxLatPort)
            StdLat   = random.randint(0, MaxLatPort)

            StAddrId = random.randint(0, NumAddr-1)
            StAddr   = AddrList[StAddrId]
            StData   = random.randint(0, MaxValue-1)
            MemValue[StAddrId] = StData

            file_sta_2.write(f'{StaLat} {StAddr}\n')
            file_std_2.write(f'{StdLat} {StData}\n')

            # load 2
            LdaLat   = random.randint(0, MaxLatPort)
            LddLat   = random.randint(0, MaxLatPort)

            LdAddrId = random.randint(0, NumAddr-1)
            LdAddr   = AddrList[LdAddrId]
            LdData   = MemValue[LdAddrId]

            file_lda_2.write(f'{LdaLat} {LdAddr}\n')
            file_ldd_2.write(f'{LddLat} {LdData}\n')

    for i in range(0, NumMem + 1):
        RamLat = random.randint(0, 1)
        file_ram.write(f'{RamLat}\n')

    file_init.close()
    file_sta_0.close()
    file_std_0.close()
    file_lda_0.close()
    file_ldd_0.close()
    file_sta_1.close()
    file_std_1.close()
    file_lda_1.close()
    file_ldd_1.close()
    file_sta_2.close()
    file_std_2.close()
    file_lda_2.close()
    file_ldd_2.close()

if __name__ == '__main__':
    gen_g2p6()
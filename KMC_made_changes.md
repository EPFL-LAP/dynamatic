# middle-end
- Added new attribute "frequency" to all arith units in HandshakeArithOps.td

-NOW : 
change JSON to include the info
change the readfromJSON -> chnages timingDB because that defines how the parsing is done; I expect this can be done by exclusively expanding the timing model

-> began tentative changes, but it seems so changes to reading logic may be required. hopefully we weon't need to chnage the objects from json.h 


# backend

looking at rtl-config-vhdl-beta.json, seems like bitwidth has been extracted and passed; if we figure out how, as long as we maintain bitwidth/frequency parallelism we ought to be able to do the same


FIGURE OUT HOW TF BITWIDTH IS OBTAINED : getHandshakeTypeBitWidth (bufferplacmeentmilp.cpp) -> getHandshakeTypeBitWidth (HandshakesTypes.cpp) -> getDataBitWidth ->  getIntOrFloatBitWidth (Types.cpp) -> builtintypes.cpp


seems bitwidth is read from channel in BufferplacementMLIR, maybe add targetCP to the channel variables? which could then pass it to addChannelPathConstraints-> getTotalDataDelay, 


possible downwards pass for targetCP:

runDynmaticPass -> placeBuffers -> getBufferPlacement -> checkLoggerAndSolve -> LEAVING HandshakePlaceBuffers solveMILP 
                                                      \> LEAVING HandshakePlaceBuffers  FPL22Buffers -> BufferPlacementMILP ->




NVM, it goes down to fgpa22, as TargetPeriod.


RESUME AT  BufferPlacementMILP::initialize()

however, timing model, in which there are the getlatency stuff, does NOT have it. so whee do we go from one to the other?

My ingress prob is line 168 bufferplacementMILP.h

-> do a draft tonight
-> figure out how on earth it can be tested, maybe write to some arbitrary file from within the function, during the run?

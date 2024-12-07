-- MarkMemoryInterfaces --> CF Transformation

sets `dynamatic::handshake::MemInterfaceAttr`

uses MemDependenceArrayAttr
just WAW dependencies of the same operation on itself are ignored. besides that, every thing is 


--  HandshakeReplaceMemoryInterfaces
handshake::MemInterfaceAttr


-- HandshakeAnalyzeLSQUsage

extracts two different sets of 
dependent load and dependent store.
based on wether it is in the dependent set it
sets MemInterfaceAttr



# Out-of-Order Execution Operations
The out-of-order execution algorithm requires 4 operations:

- FreeTagsFifo
- Tagger
- Untagger
- Aligner

## FreeTagsFifo
A `FreeTagsFifo` represents a FIFO queue that supplies tags ready for reuse.

<img alt="FreeTagsFifo diagram" src="./Figures/FreeTagsFifo.png" width="200" />

## Tagger
A `Tagger` attaches extra tag bits to the data payload.

<img alt="Tagger diagram" src="./Figures/Tagger.png" width="200" />

## Untagger
An `Untagger` extracts the extra tag bits previously attached to a data payload by the tagger.

<img alt="Untagger diagram" src="./Figures/Untagger.png" width="200" />

## Aligner
An `Aligner` synchronizes and matches tagged tokens from multiple inputs, passing them to the output once identical tags are found.

<img alt="Aligner diagram" src="./Figures/Aligner.png" width="600" />

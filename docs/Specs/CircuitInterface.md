# Circuit & Memory Interface

> [!NOTE]
> This is a proposed design change; it is not implemented yet.

The interface of Dynamatic-generated circuits has so far never been properly formalized; it is unclear what guarantees our circuits provide to the outside world, or even what the semantics of their top-level IO are. This design proposal aims to clarify these concerns and lay out clear invariants that all Dynamatic circuits must honor to allow their useas part of larger arbitrary circuits and the composition of multiple Dynamatic circuits together. This specification introduces the proposed interfaces by looking at our circuits at different levels of granularity.

1. [Circuit interface](#circuit-interface) | Describes the semantics of our circuit's top-level IO.
2. [Memory interface](#memory-interface) | Explains how we can mplement standardized memory interfaces (e.g., AXI) from our ad-hoc ones.
3. [Internal implementation example](#internal-implementation) | Example of how we may implement the circuits' semantics internally.

## Circuit interface

![Opaque circuit](figs/circuit_opaque.svg)

## Memory interface

![Transparent circuit](figs/circuit_wrapper.svg)

## Internal implementation

![Transparent circuit](figs/circuit_transparent.svg)

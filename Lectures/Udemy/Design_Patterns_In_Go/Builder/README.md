# Builder
**When piecewise object construction is complicated, provide an API for doing it succinctly.**

## Motivation
 - Some objects require a lot of ceremony to create. Such as having an object with 10 constructor arguments is not productive.
 - Builder provides an API for constructing an object step-by-step(piece-by-piece construction).

## Summary
 - A builder is a separate component for building an object.
 - To make a builder fluent, return the receiver(pointer)  - allows chaining.
 - different facets of an object can be built with different builders working together via a common struct.

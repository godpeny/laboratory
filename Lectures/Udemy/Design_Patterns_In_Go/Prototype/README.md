# Prototype
**A Partially or fully initialized object that you copy(clone) and make use of.**

## Motivation
- Complicated object aren't designed from scratch.
  - They reiterate existing designs.
- An existing (partially or fully constructed) design is a Prototype.
- We make a copy(clone) the prototype and customize it.
  - Requires 'deep copy' support. 
- We make the cloning convenient (e.g. via a Factory).

## Summary
- To implement a prototype, partially construct an object and store it somewhere.
- Deep copy the prototype.
- Customize the resulting instance.
- A prototype factory provides a convenient API for using prototypes.
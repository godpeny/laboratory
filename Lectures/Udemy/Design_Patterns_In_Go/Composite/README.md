# Composite
**A Mechanism for treating individual (scalar) objects and compositions of objects in a uniform manner.**

## Motivation
- Objects use other objects' fields/properties/members through embedding.
- Composition lets us make compound objects.
  - E.g. a mathematical expression composed of simple expressions; or
  - A shape group made of several different shapes.
- Composite design pattern is used to treat both single (scalar) and composite objects uniformly.
  - E.e. Foo and []Foo have common APIs.

## Summary
- Objects can use other objects via composition.
- Some composed and singular objects need similar/identical behaviors.
- Composite design pattern lets us treat both types of objects uniformly.
- Iteration supported with the `Iterator` design pattern.
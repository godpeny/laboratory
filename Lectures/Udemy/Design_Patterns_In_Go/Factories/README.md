# Factory
**A Component responsible solely for the wholesale (not piecewise) creation of objects.**

## Motivation
 - Object creation logic becomes too complicated.
 - Struct has too many fields, need to initialize all correctly.
 - Wholesale object creation (non-piecewise, unlike builder) can be outsourced to:
   - A separate method (Factory Method)
   - That may exist in a separate class (Factory)

## Summary
 - A factory function (a.k.a constructor) is a helper function for making struct instances.
 - A factory is any entity that can take care of object creation.
 - A factory can be a function or a dedicated struct.
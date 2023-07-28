# Decorator
**Facilitates the addition of behaviors to individual objects through embeddings**

## Motivation
- Want to augment an object with additional functionality.
- Do not want to rewrite or alter existing code (OCP).
- Want to keep new functionality separate (SRP).
- Need to be able to interact with existing structures.
- Solution: embed the decorated object and provide additional functionality.

## Summary
- A decorator embeds decorated objects.
- Adds utility fields and methods to augment the object's features.
- Often used to emulate multiple inheritance (may require extra work, e.g. multiple_aggregation).
# Singleton
**A component which is instantiated only once.**

## Motivation
- For some components it only makes sense to have one in the system.
  - Database repository.
  - Object factory.
- E.g. the constructor call is expensive.
  - We only do it once.
  - We provide everyone with the same instance.
- Want to prevent anyone creating additional copies.
- Need to take care of lazy instantiation and thread safety.

## Summary
- Lazy one-time initialization using sync.Once.
- Adhere to DIP: depend on interfaces, not concrete types.
- Singleton is not scary, but be careful with how you use it.
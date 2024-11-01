# Adapter
**A construct which adapts an existing interface X to conform to the required interface Y.**

## Motivation
- Electrical devices have different power(interface) requirements.
  - Voltage (110V vs. 220V).
  - Socket/plug type. (US, EU, UK ...)
- We cannot modify our gadgets to support every possible interface.
- Thus: use a special device (an adapter) to convert to the required interface.

## Summary
- Implementing the adapter pattern is easy.
- Determine the API you have and the API you need.
- Create a component which aggregates (has a reference to, ...) the adaptee.
- Intermediate representations can pile up: use caching and other optimizations.
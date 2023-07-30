# Chain Of Responsibility
**A Chain of components who all get a chance to process a command or a query, optionally having default processing implementation and an ability to terminate the processing chain.**

## Motivation
- Unethical behavior by an employee; who takes the blame?
  - Employee
  - Manager
  - CEO
- You click a graphical element on a form.
  - Button handles it, stops further processing
  - Underlying group box
  - Underlying window
- CCG computer game
  - Creature has attack and defense values
  - Those can be boosted by other cards

## Command And Query Separation
**https://en.wikipedia.org/wiki/Command%E2%80%93query_separation**
- Command = asking for an action or change (e.g., please set your attack value to 2)
- Query = asking for information (e.g., please give me your attack value)
- CQS = having separate means of sending commands and queries to e.g. direct field access

## Summary
- Chain of Responsibility can be implemented as a chain of references or a centralized construct (observer pattern)
- Enlist objects in the chain, possibly controlling their order
- Control object removal from chain

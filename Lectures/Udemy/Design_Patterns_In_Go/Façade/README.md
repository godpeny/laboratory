# Façade
**Provides a simple, easy to understand/user interface over a large and sophisticated body of code.**

## Motivation
- Balancing complexity and presentation/usability.
- Typical home: 
  - many subsystems (electrical sanitation).
  - complex internal structure (e.g. floor layers).
  - end user is not exposed to internals.
- Same with software!
  - Many systems working to provide flexibility, but...
  - API consumers want it to 'just work'.

## Summary
- Build a Façade to provide a simplified API over a set of components
- May wish to (optionally) expose internals through the façade
- May allow users to 'escalate' to use more complex APIs if they need to
- 
# Flyweight
**A Space optimization technique that lets us use less memory by storing externally the data associated with similar objects**

## Motivation
- Avoid redundancy when storing data
- E.g. MMORPG
  - Plenty of users with identical first/last names
  - No sense in storing same first/last name over and over again
  - Store a list of names and pointers to them
- E.G., bold or italic text in a document
  - Don't want each character to have a formatting character
  - Operate on ranges (e.g., line number, start/end positions)
  
## Summary
- Store common data externally
- Specify an index or a reference into the external data store
- Define the idea of 'ranges' on homogeneous collections and store data related to those ranges

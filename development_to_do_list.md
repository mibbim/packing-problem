# Refactoring :


- Maybe a Class Problem should be implemented,
that contains the problem data (l, h, R...)
- Maybe a Class Circle should be implemented, that implements geometrical
properties. Computing Areas, sagittas, etc. should be their responsability.
- Maybe reduce the number of properties in the classes OPP.
- Refactor so that add_variables returns a dictionary of variables instead of a tuples to make code more robust and mantainable.
- Refactor entangling Opp and Opp_rot with composition instead of inheritance.
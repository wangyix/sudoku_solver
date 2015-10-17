This is my final project for the Stanford Spring 2015 class CS232 / EE368.
The project is an Android app that uses the mobile device's camera to
automatically solve any Sudoku puzzle.  The detected values and the solution
values are overlaid onto the camera image in (almost) real-time.

In order for the puzzle to be successfully detected, the following conditions
need to be fulfilled:
- The puzzle grid and digits should be dark on a light background.
- The four sides of the puzzle must at least have a thin border of light
  background surrouding it.
- The four outer corners of the puzzle must all be visible in the camera image.

Blue digits should appear over the digits that are already filled in and should
match the actual printed digits in the puzzle.  Green digits should appear over
the empty cells of the puzzle indicating the solution values.  If a solution
isn't found (usually caused by a printed digit being unrecognized or
misrecognized, or if the camera image does not contain a Sudoku puzzlie),
then 81 small green dots should appear.  These represent where the app believes
the centers of the 81 cells of the Sudoku grid are.  This was a handy method
for debugging the app while I was working on it.
 
# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake.exe

# The command to remove a file.
RM = /usr/bin/cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build

# Include any dependencies generated for this target.
include bench/spbench/CMakeFiles/spbenchsolver.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bench/spbench/CMakeFiles/spbenchsolver.dir/compiler_depend.make

# Include the progress variables for this target.
include bench/spbench/CMakeFiles/spbenchsolver.dir/progress.make

# Include the compile flags for this target's objects.
include bench/spbench/CMakeFiles/spbenchsolver.dir/flags.make

bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o: bench/spbench/CMakeFiles/spbenchsolver.dir/flags.make
bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/bench/spbench/spbenchsolver.cpp
bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o: bench/spbench/CMakeFiles/spbenchsolver.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o -MF CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o.d -o CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/bench/spbench/spbenchsolver.cpp

bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/bench/spbench/spbenchsolver.cpp > CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.i

bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/bench/spbench/spbenchsolver.cpp -o CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.s

# Object files for target spbenchsolver
spbenchsolver_OBJECTS = \
"CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o"

# External object files for target spbenchsolver
spbenchsolver_EXTERNAL_OBJECTS =

bench/spbench/spbenchsolver.exe: bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o
bench/spbench/spbenchsolver.exe: bench/spbench/CMakeFiles/spbenchsolver.dir/build.make
bench/spbench/spbenchsolver.exe: /usr/lib/librt.a
bench/spbench/spbenchsolver.exe: bench/spbench/CMakeFiles/spbenchsolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable spbenchsolver.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spbenchsolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bench/spbench/CMakeFiles/spbenchsolver.dir/build: bench/spbench/spbenchsolver.exe
.PHONY : bench/spbench/CMakeFiles/spbenchsolver.dir/build

bench/spbench/CMakeFiles/spbenchsolver.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench && $(CMAKE_COMMAND) -P CMakeFiles/spbenchsolver.dir/cmake_clean.cmake
.PHONY : bench/spbench/CMakeFiles/spbenchsolver.dir/clean

bench/spbench/CMakeFiles/spbenchsolver.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/bench/spbench /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bench/spbench/CMakeFiles/spbenchsolver.dir/depend


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
CMAKE_SOURCE_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build

# Utility rule file for blas.

# Include any custom commands dependencies for this target.
include blas/CMakeFiles/blas.dir/compiler_depend.make

# Include the progress variables for this target.
include blas/CMakeFiles/blas.dir/progress.make

blas: blas/CMakeFiles/blas.dir/build.make
.PHONY : blas

# Rule to build all files generated by this target.
blas/CMakeFiles/blas.dir/build: blas
.PHONY : blas/CMakeFiles/blas.dir/build

blas/CMakeFiles/blas.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/blas && $(CMAKE_COMMAND) -P CMakeFiles/blas.dir/cmake_clean.cmake
.PHONY : blas/CMakeFiles/blas.dir/clean

blas/CMakeFiles/blas.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen/blas /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/blas /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/blas/CMakeFiles/blas.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : blas/CMakeFiles/blas.dir/depend


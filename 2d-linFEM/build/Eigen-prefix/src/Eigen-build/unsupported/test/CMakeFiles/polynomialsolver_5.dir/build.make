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

# Include any dependencies generated for this target.
include unsupported/test/CMakeFiles/polynomialsolver_5.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include unsupported/test/CMakeFiles/polynomialsolver_5.dir/compiler_depend.make

# Include the progress variables for this target.
include unsupported/test/CMakeFiles/polynomialsolver_5.dir/progress.make

# Include the compile flags for this target's objects.
include unsupported/test/CMakeFiles/polynomialsolver_5.dir/flags.make

unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o: unsupported/test/CMakeFiles/polynomialsolver_5.dir/flags.make
unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen/unsupported/test/polynomialsolver.cpp
unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o: unsupported/test/CMakeFiles/polynomialsolver_5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o -MF CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o.d -o CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen/unsupported/test/polynomialsolver.cpp

unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen/unsupported/test/polynomialsolver.cpp > CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.i

unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen/unsupported/test/polynomialsolver.cpp -o CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.s

# Object files for target polynomialsolver_5
polynomialsolver_5_OBJECTS = \
"CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o"

# External object files for target polynomialsolver_5
polynomialsolver_5_EXTERNAL_OBJECTS =

unsupported/test/polynomialsolver_5.exe: unsupported/test/CMakeFiles/polynomialsolver_5.dir/polynomialsolver.cpp.o
unsupported/test/polynomialsolver_5.exe: unsupported/test/CMakeFiles/polynomialsolver_5.dir/build.make
unsupported/test/polynomialsolver_5.exe: unsupported/test/CMakeFiles/polynomialsolver_5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable polynomialsolver_5.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/polynomialsolver_5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unsupported/test/CMakeFiles/polynomialsolver_5.dir/build: unsupported/test/polynomialsolver_5.exe
.PHONY : unsupported/test/CMakeFiles/polynomialsolver_5.dir/build

unsupported/test/CMakeFiles/polynomialsolver_5.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test && $(CMAKE_COMMAND) -P CMakeFiles/polynomialsolver_5.dir/cmake_clean.cmake
.PHONY : unsupported/test/CMakeFiles/polynomialsolver_5.dir/clean

unsupported/test/CMakeFiles/polynomialsolver_5.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen/unsupported/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-linFEM/build/Eigen-prefix/src/Eigen-build/unsupported/test/CMakeFiles/polynomialsolver_5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unsupported/test/CMakeFiles/polynomialsolver_5.dir/depend


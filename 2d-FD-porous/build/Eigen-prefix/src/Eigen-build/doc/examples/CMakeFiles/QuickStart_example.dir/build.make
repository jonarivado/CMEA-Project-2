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
include doc/examples/CMakeFiles/QuickStart_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include doc/examples/CMakeFiles/QuickStart_example.dir/compiler_depend.make

# Include the progress variables for this target.
include doc/examples/CMakeFiles/QuickStart_example.dir/progress.make

# Include the compile flags for this target's objects.
include doc/examples/CMakeFiles/QuickStart_example.dir/flags.make

doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o: doc/examples/CMakeFiles/QuickStart_example.dir/flags.make
doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/doc/examples/QuickStart_example.cpp
doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o: doc/examples/CMakeFiles/QuickStart_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o -MF CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o.d -o CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/doc/examples/QuickStart_example.cpp

doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/doc/examples/QuickStart_example.cpp > CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.i

doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/doc/examples/QuickStart_example.cpp -o CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.s

# Object files for target QuickStart_example
QuickStart_example_OBJECTS = \
"CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o"

# External object files for target QuickStart_example
QuickStart_example_EXTERNAL_OBJECTS =

doc/examples/QuickStart_example.exe: doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o
doc/examples/QuickStart_example.exe: doc/examples/CMakeFiles/QuickStart_example.dir/build.make
doc/examples/QuickStart_example.exe: doc/examples/CMakeFiles/QuickStart_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable QuickStart_example.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/QuickStart_example.dir/link.txt --verbose=$(VERBOSE)
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples && ./QuickStart_example.exe >/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples/QuickStart_example.out

# Rule to build all files generated by this target.
doc/examples/CMakeFiles/QuickStart_example.dir/build: doc/examples/QuickStart_example.exe
.PHONY : doc/examples/CMakeFiles/QuickStart_example.dir/build

doc/examples/CMakeFiles/QuickStart_example.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/QuickStart_example.dir/cmake_clean.cmake
.PHONY : doc/examples/CMakeFiles/QuickStart_example.dir/clean

doc/examples/CMakeFiles/QuickStart_example.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/doc/examples /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/doc/examples/CMakeFiles/QuickStart_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/examples/CMakeFiles/QuickStart_example.dir/depend


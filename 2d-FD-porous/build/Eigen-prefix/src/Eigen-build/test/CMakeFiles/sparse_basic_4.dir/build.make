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
include test/CMakeFiles/sparse_basic_4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/sparse_basic_4.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/sparse_basic_4.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/sparse_basic_4.dir/flags.make

test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o: test/CMakeFiles/sparse_basic_4.dir/flags.make
test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/sparse_basic.cpp
test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o: test/CMakeFiles/sparse_basic_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o -MF CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o.d -o CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/sparse_basic.cpp

test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/sparse_basic.cpp > CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.i

test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/sparse_basic.cpp -o CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.s

# Object files for target sparse_basic_4
sparse_basic_4_OBJECTS = \
"CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o"

# External object files for target sparse_basic_4
sparse_basic_4_EXTERNAL_OBJECTS =

test/sparse_basic_4.exe: test/CMakeFiles/sparse_basic_4.dir/sparse_basic.cpp.o
test/sparse_basic_4.exe: test/CMakeFiles/sparse_basic_4.dir/build.make
test/sparse_basic_4.exe: test/CMakeFiles/sparse_basic_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sparse_basic_4.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sparse_basic_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/sparse_basic_4.dir/build: test/sparse_basic_4.exe
.PHONY : test/CMakeFiles/sparse_basic_4.dir/build

test/CMakeFiles/sparse_basic_4.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/sparse_basic_4.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/sparse_basic_4.dir/clean

test/CMakeFiles/sparse_basic_4.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test/CMakeFiles/sparse_basic_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/sparse_basic_4.dir/depend


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
include test/CMakeFiles/mapstride_2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/mapstride_2.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/mapstride_2.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/mapstride_2.dir/flags.make

test/CMakeFiles/mapstride_2.dir/mapstride.cpp.o: test/CMakeFiles/mapstride_2.dir/flags.make
test/CMakeFiles/mapstride_2.dir/mapstride.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/mapstride.cpp
test/CMakeFiles/mapstride_2.dir/mapstride.cpp.o: test/CMakeFiles/mapstride_2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/mapstride_2.dir/mapstride.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/mapstride_2.dir/mapstride.cpp.o -MF CMakeFiles/mapstride_2.dir/mapstride.cpp.o.d -o CMakeFiles/mapstride_2.dir/mapstride.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/mapstride.cpp

test/CMakeFiles/mapstride_2.dir/mapstride.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mapstride_2.dir/mapstride.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/mapstride.cpp > CMakeFiles/mapstride_2.dir/mapstride.cpp.i

test/CMakeFiles/mapstride_2.dir/mapstride.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mapstride_2.dir/mapstride.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test/mapstride.cpp -o CMakeFiles/mapstride_2.dir/mapstride.cpp.s

# Object files for target mapstride_2
mapstride_2_OBJECTS = \
"CMakeFiles/mapstride_2.dir/mapstride.cpp.o"

# External object files for target mapstride_2
mapstride_2_EXTERNAL_OBJECTS =

test/mapstride_2.exe: test/CMakeFiles/mapstride_2.dir/mapstride.cpp.o
test/mapstride_2.exe: test/CMakeFiles/mapstride_2.dir/build.make
test/mapstride_2.exe: test/CMakeFiles/mapstride_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mapstride_2.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mapstride_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/mapstride_2.dir/build: test/mapstride_2.exe
.PHONY : test/CMakeFiles/mapstride_2.dir/build

test/CMakeFiles/mapstride_2.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/mapstride_2.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/mapstride_2.dir/clean

test/CMakeFiles/mapstride_2.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/test/CMakeFiles/mapstride_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/mapstride_2.dir/depend


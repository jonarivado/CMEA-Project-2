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
CMAKE_SOURCE_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build

# Include any dependencies generated for this target.
include CMakeFiles/create_poisson_matrix.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/create_poisson_matrix.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/create_poisson_matrix.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/create_poisson_matrix.dir/flags.make

CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o: CMakeFiles/create_poisson_matrix.dir/flags.make
CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o: ../create_poisson_matrix.cpp
CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o: CMakeFiles/create_poisson_matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o -MF CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o.d -o CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/create_poisson_matrix.cpp

CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.i"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/create_poisson_matrix.cpp > CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.i

CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.s"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/create_poisson_matrix.cpp -o CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.s

# Object files for target create_poisson_matrix
create_poisson_matrix_OBJECTS = \
"CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o"

# External object files for target create_poisson_matrix
create_poisson_matrix_EXTERNAL_OBJECTS =

libcreate_poisson_matrix.a: CMakeFiles/create_poisson_matrix.dir/create_poisson_matrix.cpp.o
libcreate_poisson_matrix.a: CMakeFiles/create_poisson_matrix.dir/build.make
libcreate_poisson_matrix.a: CMakeFiles/create_poisson_matrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcreate_poisson_matrix.a"
	$(CMAKE_COMMAND) -P CMakeFiles/create_poisson_matrix.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/create_poisson_matrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/create_poisson_matrix.dir/build: libcreate_poisson_matrix.a
.PHONY : CMakeFiles/create_poisson_matrix.dir/build

CMakeFiles/create_poisson_matrix.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/create_poisson_matrix.dir/cmake_clean.cmake
.PHONY : CMakeFiles/create_poisson_matrix.dir/clean

CMakeFiles/create_poisson_matrix.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/CMakeFiles/create_poisson_matrix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/create_poisson_matrix.dir/depend


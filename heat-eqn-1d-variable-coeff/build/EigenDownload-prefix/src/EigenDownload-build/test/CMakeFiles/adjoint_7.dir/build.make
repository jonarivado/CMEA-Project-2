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
CMAKE_SOURCE_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build

# Include any dependencies generated for this target.
include test/CMakeFiles/adjoint_7.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/adjoint_7.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/adjoint_7.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/adjoint_7.dir/flags.make

test/CMakeFiles/adjoint_7.dir/adjoint.cpp.o: test/CMakeFiles/adjoint_7.dir/flags.make
test/CMakeFiles/adjoint_7.dir/adjoint.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/test/adjoint.cpp
test/CMakeFiles/adjoint_7.dir/adjoint.cpp.o: test/CMakeFiles/adjoint_7.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/adjoint_7.dir/adjoint.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/adjoint_7.dir/adjoint.cpp.o -MF CMakeFiles/adjoint_7.dir/adjoint.cpp.o.d -o CMakeFiles/adjoint_7.dir/adjoint.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/test/adjoint.cpp

test/CMakeFiles/adjoint_7.dir/adjoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/adjoint_7.dir/adjoint.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/test/adjoint.cpp > CMakeFiles/adjoint_7.dir/adjoint.cpp.i

test/CMakeFiles/adjoint_7.dir/adjoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/adjoint_7.dir/adjoint.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/test/adjoint.cpp -o CMakeFiles/adjoint_7.dir/adjoint.cpp.s

# Object files for target adjoint_7
adjoint_7_OBJECTS = \
"CMakeFiles/adjoint_7.dir/adjoint.cpp.o"

# External object files for target adjoint_7
adjoint_7_EXTERNAL_OBJECTS =

test/adjoint_7.exe: test/CMakeFiles/adjoint_7.dir/adjoint.cpp.o
test/adjoint_7.exe: test/CMakeFiles/adjoint_7.dir/build.make
test/adjoint_7.exe: test/CMakeFiles/adjoint_7.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable adjoint_7.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/adjoint_7.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/adjoint_7.dir/build: test/adjoint_7.exe
.PHONY : test/CMakeFiles/adjoint_7.dir/build

test/CMakeFiles/adjoint_7.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test && $(CMAKE_COMMAND) -P CMakeFiles/adjoint_7.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/adjoint_7.dir/clean

test/CMakeFiles/adjoint_7.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/test/CMakeFiles/adjoint_7.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/adjoint_7.dir/depend


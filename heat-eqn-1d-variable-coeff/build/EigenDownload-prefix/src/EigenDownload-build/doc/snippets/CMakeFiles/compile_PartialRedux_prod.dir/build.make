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
include doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compiler_depend.make

# Include the progress variables for this target.
include doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/progress.make

# Include the compile flags for this target's objects.
include doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/flags.make

doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o: doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/flags.make
doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o: doc/snippets/compile_PartialRedux_prod.cpp
doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/doc/snippets/PartialRedux_prod.cpp
doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o: doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o -MF CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o.d -o CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets/compile_PartialRedux_prod.cpp

doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets/compile_PartialRedux_prod.cpp > CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.i

doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets/compile_PartialRedux_prod.cpp -o CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.s

# Object files for target compile_PartialRedux_prod
compile_PartialRedux_prod_OBJECTS = \
"CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o"

# External object files for target compile_PartialRedux_prod
compile_PartialRedux_prod_EXTERNAL_OBJECTS =

doc/snippets/compile_PartialRedux_prod.exe: doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/compile_PartialRedux_prod.cpp.o
doc/snippets/compile_PartialRedux_prod.exe: doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/build.make
doc/snippets/compile_PartialRedux_prod.exe: doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compile_PartialRedux_prod.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_PartialRedux_prod.dir/link.txt --verbose=$(VERBOSE)
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets && ./compile_PartialRedux_prod.exe >/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets/PartialRedux_prod.out

# Rule to build all files generated by this target.
doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/build: doc/snippets/compile_PartialRedux_prod.exe
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/build

doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_PartialRedux_prod.dir/cmake_clean.cmake
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/clean

doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/Eigen/doc/snippets /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/heat-eqn-1d-variable-coeff/build/EigenDownload-prefix/src/EigenDownload-build/doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_prod.dir/depend


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
include unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/compiler_depend.make

# Include the progress variables for this target.
include unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/progress.make

# Include the compile flags for this target's objects.
include unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/flags.make

unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o: unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/flags.make
unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o: /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/unsupported/test/cxx11_tensor_casts.cpp
unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o: unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o -MF CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o.d -o CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o -c /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/unsupported/test/cxx11_tensor_casts.cpp

unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.i"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/unsupported/test/cxx11_tensor_casts.cpp > CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.i

unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.s"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test && /usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/unsupported/test/cxx11_tensor_casts.cpp -o CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.s

# Object files for target cxx11_tensor_casts
cxx11_tensor_casts_OBJECTS = \
"CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o"

# External object files for target cxx11_tensor_casts
cxx11_tensor_casts_EXTERNAL_OBJECTS =

unsupported/test/cxx11_tensor_casts.exe: unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/cxx11_tensor_casts.cpp.o
unsupported/test/cxx11_tensor_casts.exe: unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/build.make
unsupported/test/cxx11_tensor_casts.exe: unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cxx11_tensor_casts.exe"
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cxx11_tensor_casts.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/build: unsupported/test/cxx11_tensor_casts.exe
.PHONY : unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/build

unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/clean:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test && $(CMAKE_COMMAND) -P CMakeFiles/cxx11_tensor_casts.dir/cmake_clean.cmake
.PHONY : unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/clean

unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/depend:
	cd /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen/unsupported/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test /cygdrive/c/Users/jonat/OneDrive/Documents/CMEA/project2_handout/2d-FD-porous/build/Eigen-prefix/src/Eigen-build/unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unsupported/test/CMakeFiles/cxx11_tensor_casts.dir/depend


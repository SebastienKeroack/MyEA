# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2019.1.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2019.1.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Neural_Network_Library_Linux.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Neural_Network_Library_Linux.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Neural_Network_Library_Linux.dir/flags.make

CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.o: CMakeFiles/Neural_Network_Library_Linux.dir/flags.make
CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.o: ../src/stdafx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.o -c /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/src/stdafx.cpp

CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/src/stdafx.cpp > CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.i

CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/src/stdafx.cpp -o CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.s

# Object files for target Neural_Network_Library_Linux
Neural_Network_Library_Linux_OBJECTS = \
"CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.o"

# External object files for target Neural_Network_Library_Linux
Neural_Network_Library_Linux_EXTERNAL_OBJECTS =

libNeural_Network_Library_Linux.so: CMakeFiles/Neural_Network_Library_Linux.dir/src/stdafx.cpp.o
libNeural_Network_Library_Linux.so: CMakeFiles/Neural_Network_Library_Linux.dir/build.make
libNeural_Network_Library_Linux.so: CMakeFiles/Neural_Network_Library_Linux.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libNeural_Network_Library_Linux.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Neural_Network_Library_Linux.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Neural_Network_Library_Linux.dir/build: libNeural_Network_Library_Linux.so

.PHONY : CMakeFiles/Neural_Network_Library_Linux.dir/build

CMakeFiles/Neural_Network_Library_Linux.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Neural_Network_Library_Linux.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Neural_Network_Library_Linux.dir/clean

CMakeFiles/Neural_Network_Library_Linux.dir/depend:
	cd /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug /home/sebastien/MEGAsync/MyEA/MyEA_Visual_Studio/Neural_Network_Library_Linux/cmake-build-debug/CMakeFiles/Neural_Network_Library_Linux.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Neural_Network_Library_Linux.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp

# Include any dependencies generated for this target.
include CMakeFiles/SVO_Recording.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SVO_Recording.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SVO_Recording.dir/flags.make

CMakeFiles/SVO_Recording.dir/main.o: CMakeFiles/SVO_Recording.dir/flags.make
CMakeFiles/SVO_Recording.dir/main.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SVO_Recording.dir/main.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SVO_Recording.dir/main.o -c /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp/main.cpp

CMakeFiles/SVO_Recording.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SVO_Recording.dir/main.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp/main.cpp > CMakeFiles/SVO_Recording.dir/main.i

CMakeFiles/SVO_Recording.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SVO_Recording.dir/main.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp/main.cpp -o CMakeFiles/SVO_Recording.dir/main.s

CMakeFiles/SVO_Recording.dir/main.o.requires:

.PHONY : CMakeFiles/SVO_Recording.dir/main.o.requires

CMakeFiles/SVO_Recording.dir/main.o.provides: CMakeFiles/SVO_Recording.dir/main.o.requires
	$(MAKE) -f CMakeFiles/SVO_Recording.dir/build.make CMakeFiles/SVO_Recording.dir/main.o.provides.build
.PHONY : CMakeFiles/SVO_Recording.dir/main.o.provides

CMakeFiles/SVO_Recording.dir/main.o.provides.build: CMakeFiles/SVO_Recording.dir/main.o


# Object files for target SVO_Recording
SVO_Recording_OBJECTS = \
"CMakeFiles/SVO_Recording.dir/main.o"

# External object files for target SVO_Recording
SVO_Recording_EXTERNAL_OBJECTS =

SVO_Recording: CMakeFiles/SVO_Recording.dir/main.o
SVO_Recording: CMakeFiles/SVO_Recording.dir/build.make
SVO_Recording: /usr/local/zed/lib/libsl_zed.so
SVO_Recording: /usr/lib/aarch64-linux-gnu/libopenblas.so
SVO_Recording: /usr/lib/aarch64-linux-gnu/libusb-1.0.so
SVO_Recording: /usr/lib/aarch64-linux-gnu/libcuda.so
SVO_Recording: /usr/local/cuda-10.2/lib64/libcudart.so
SVO_Recording: CMakeFiles/SVO_Recording.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SVO_Recording"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SVO_Recording.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SVO_Recording.dir/build: SVO_Recording

.PHONY : CMakeFiles/SVO_Recording.dir/build

CMakeFiles/SVO_Recording.dir/requires: CMakeFiles/SVO_Recording.dir/main.o.requires

.PHONY : CMakeFiles/SVO_Recording.dir/requires

CMakeFiles/SVO_Recording.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SVO_Recording.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SVO_Recording.dir/clean

CMakeFiles/SVO_Recording.dir/depend:
	cd /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp /home/smartwater/Documentos/smartwater/Tensorflow/scripts/svo_recording/cpp/CMakeFiles/SVO_Recording.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SVO_Recording.dir/depend

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/flags.make

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/flags.make
CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o -c /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/main.cpp

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/main.cpp > CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.i

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/main.cpp -o CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.s

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.requires

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.provides: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/build.make CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.provides

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.provides.build: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o


# Object files for target HW05_Shah_Niyati_PartAandB
HW05_Shah_Niyati_PartAandB_OBJECTS = \
"CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o"

# External object files for target HW05_Shah_Niyati_PartAandB
HW05_Shah_Niyati_PartAandB_EXTERNAL_OBJECTS =

HW05_Shah_Niyati_PartAandB: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o
HW05_Shah_Niyati_PartAandB: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/build.make
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_shape.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_stitching.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_superres.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_videostab.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_objdetect.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_calib3d.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_features2d.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_flann.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_highgui.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_ml.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_photo.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_video.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_videoio.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_imgcodecs.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_imgproc.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: /usr/local/lib/libopencv_core.3.2.0.dylib
HW05_Shah_Niyati_PartAandB: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable HW05_Shah_Niyati_PartAandB"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/build: HW05_Shah_Niyati_PartAandB

.PHONY : CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/build

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/requires: CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/main.cpp.o.requires

.PHONY : CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/requires

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/cmake_clean.cmake
.PHONY : CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/clean

CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/depend:
	cd /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug /Users/niyatishah/CLionProjects/HW05_Shah_Niyati_PartAandB/cmake-build-debug/CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/HW05_Shah_Niyati_PartAandB.dir/depend

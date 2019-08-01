CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(Common_Library_Linux VERSION 1.0.0 DESCRIPTION "No description.")

ADD_DEFINITIONS(-DCOMPILE_x64
                -DCOMPILE_DOUBLE
                -DCOMPILE_COUT
                -DCOMPILE_LINUX
                -DCOMPILE_UINPUT
                -DCOMPILE_BREAK_CODE
                -DCOMPILE_DEBUG)

SET(CMAKE_CXX_STANDARD 17)

# |STR| Library. |STR|
INCLUDE_DIRECTORIES(include)

SET(SOURCE_FILES
    src/stdafx.cpp
    src/Configuration/Configuration.cpp
    src/Configuration/Linux/Linux__Configuration.cpp
    src/Files/File.cpp
    src/Keyboard_Capturing/Linux/Linux__Keyboard_Capturing.cpp
    src/Math/CPU/Math.cpp
    src/Math/MODWT.cpp
    src/Random/Random.cpp
    src/Shutdown_Capturing/Linux/Linux__Shutdown_Capturing.cpp
    src/Strings/Animation_Waiting.cpp
    src/Strings/String.cpp
    src/Time/Time.cpp
    src/UI/Message_Box.cpp)

ADD_LIBRARY(Common_Library_Linux SHARED ${SOURCE_FILES})
# |END| Library. |END|

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(Neural_Network_Library_Linux VERSION 1.0.0 DESCRIPTION "No description.")

ADD_DEFINITIONS(-DCOMPILE_x64
                -DCOMPILE_DOUBLE
                -DCOMPILE_COUT
                -DCOMPILE_WINDOWS
                -DCOMPILE_UINPUT
                -DCOMPILE_BREAK_CODE
                -DCOMPILE_DEBUG
                -DCOMPILE_ADEPT
                -DADEPT_TRACK_NON_FINITE_GRADIENTS
                -DADEPT_RECORDING_PAUSABLE)

SET(CMAKE_CXX_STANDARD 17)

# |STR| Library. |STR|
INCLUDE_DIRECTORIES(include)

SET(SOURCE_FILES
    src/stdafx.cpp)

ADD_LIBRARY(Neural_Network_Library_Linux SHARED ${SOURCE_FILES})
# |END| Library. |END|

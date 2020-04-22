CMAKE_MINIMUM_REQUIRED(VERSION 3.14.5)

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
    src/pch.cpp
    src/Capturing/Keyboard/Linux/Keyboard.cpp
    src/Capturing/Shutdown/Linux/Shutdown.cpp
    src/Configuration/Configuration.cpp
    src/Configuration/Linux/Configuration.cpp
    src/Files/File.cpp
    src/Files/Linux/File.cpp
    src/Math/MODWT.cpp
    src/Math/CPU/Math.cpp
    src/Random/Base.cpp
    src/Random/Bernoulli.cpp
    src/Random/Floating.cpp
    src/Random/Gaussian.cpp
    src/Random/Integer.cpp
    src/Strings/Animation_Waiting.cpp
    src/Strings/String.cpp
    src/Strings/Linux/String.cpp
    src/Time/Time.cpp
    src/Time/Linux/Time.cpp
    src/UI/Dialog_Box.cpp
    src/UI/Linux/Dialog_Box.cpp)

ADD_LIBRARY(Common_Library_Linux SHARED ${SOURCE_FILES})
# |END| Library. |END|

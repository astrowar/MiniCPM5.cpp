cmake_minimum_required(VERSION 3.14)
project(jinja_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include parent project headers
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${PROJECT_ROOT}/include)

add_executable(chat_template_test
    src/main.cpp
    ${PROJECT_ROOT}/src/chat_template.cpp
)

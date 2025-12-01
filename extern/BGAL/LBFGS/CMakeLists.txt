cmake_minimum_required(VERSION 3.16)
# Get static lib
add_library(LBFGS STATIC LBFGS.h)
set_target_properties(LBFGS PROPERTIES LINKER_LANGUAGE CXX)
# NOTE: need to use Eigen3::Eigen in main cmake file
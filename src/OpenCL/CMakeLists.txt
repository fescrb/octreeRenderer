include(CMakeSources.cmake)

message("sources " ${sources})
message("this_sources " ${this_sources})

foreach(source ${this_sources})
     set(sources ${sources} OpenCL/${source})
endforeach(source)

set(sources ${sources} PARENT_SCOPE)

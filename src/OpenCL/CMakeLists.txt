include(CMakeSources.cmake)

foreach(source ${this_sources})
     set(sources ${sources} OpenCL/${source})
endforeach(source)

set(sources ${sources} PARENT_SCOPE)

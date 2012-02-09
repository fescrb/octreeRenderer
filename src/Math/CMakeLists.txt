include(CMakeSources.cmake)

foreach(source ${this_sources})
     set(sources ${sources} Math/${source})
endforeach(source)

set(sources ${sources} PARENT_SCOPE)

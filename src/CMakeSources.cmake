set(sources ${sources}
			DeviceManager.cpp
			DataManager.cpp
            ProgramState.cpp
            OctreeWindow.cpp
            OctreeRendererWindow.cpp
            GeometryOctreeWindow.cpp
            )

set(headers ${headers}
			DeviceManager.h
			DataManager.h
            ProgramState.h
            OctreeWindow.h
            OctreeRendererWindow.h
            GeometryOctreeWindow.h
			)

set(shaders ${shaders}
			NoTransform.vert
			Coalesce.frag
			)

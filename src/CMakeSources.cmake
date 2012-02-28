set(sources ${sources}
			DeviceManager.cpp
			DataManager.cpp
            ProgramState.cpp
            Window.cpp
            OctreeRendererWindow.cpp
            )

set(headers ${headers}
			DeviceManager.h
			DataManager.h
            ProgramState.h
            Window.h
            OctreeRendererWindow.h
			)

set(shaders ${shaders}
			NoTransform.vert
			Coalesce.frag
			)

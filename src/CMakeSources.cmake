set(sources ${sources}
			DeviceInfo.cpp
			Device.cpp
			DeviceManager.cpp
			DataManager.cpp
            ProgramState.cpp
            Window.cpp
			main.cpp )

set(headers ${headers}
			RenderInfo.h
			DeviceInfo.h
			Device.h 
			Context.h
			DeviceManager.h
			DataManager.h
            ProgramState.h
            Window.h
			)

set(shaders ${shaders}
			NoTransform.vert
			Coalesce.frag
            RenderInfo.h
			)

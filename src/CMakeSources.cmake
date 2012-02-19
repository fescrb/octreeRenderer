set(sources ${sources}
			Image.cpp
			DeviceInfo.cpp
			Device.cpp
			DeviceManager.cpp
			DataManager.cpp
            SourceFile.cpp
            SourceFileManager.cpp
            ProgramState.cpp
            Window.cpp
			main.cpp )

set(headers ${headers}
			Image.h
			RenderInfo.h
			DeviceInfo.h
			Device.h 
			Context.h
			DeviceManager.h
			DataManager.h
            SourceFile.h
            SourceFileManager.h
            ProgramState.h
            Window.h
			)

set(shaders ${shaders}
			NoTransform.vert
			Coalesce.frag
			)

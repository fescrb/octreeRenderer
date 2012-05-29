#ifndef _DEVICE_H
#define _DEVICE_H

#include "Vector.h"
#include "Rect.h"

#include "HighResTimer.h"

#include "FramebufferWindow.h"
#include "Graphics.h"
#include "Bin.h"

#include <vector>

class DeviceInfo;
class OctreeSegment;
class renderinfo;

class Device {
    public:
		explicit                     Device();
        virtual                     ~Device();

		virtual char                *getName();

        virtual void                 printInfo() = 0;

        /**
         * We clear the framebuffer if we needen't generate it
         * @param The dimensions of the required framebuffer.
         */
        virtual void                 makeFrameBuffer(int2 size);
        virtual void                 sendData(Bin bin) = 0;
        virtual void                 sendHeader(Bin bin) = 0;
        virtual void                 renderTask(int index, renderinfo *info) = 0;
        //virtual void             render(rect *window, renderinfo *info) = 0;
        /**
         * Returns the framebuffer as a texture. NOTE: we always
         * assume that the target OpenGL context is CURRENT.
         * @return The OpenGL texture id containing the framebuffer.
         */
        virtual framebuffer_window   getFrameBuffer() = 0;
        virtual char                *getFrame() = 0;

        virtual high_res_timer       getRenderTime() = 0;
        virtual high_res_timer       getBufferToTextureTime() = 0;

        /*
         * Task-related functions
         */

        void                         clearTasks();
        void                         addTask(rect task);
        rect                        *getTask(int index);
        std::vector<rect>            getTasks();
        int                          getTaskCount();
        rect                         getTotalTaskWindow();

    protected:

        DeviceInfo                  *m_pDeviceInfo;
        
        char                        *m_pFrame;
        float                       *m_pDepthBuffer;
        unsigned char               *m_pIterations;
        unsigned char               *m_pOctreeDepth;
        
        int2                         m_frameBufferResolution;

        std::vector<rect>            m_tasks;
        rect                         m_tasksWindow;
};

#endif // _DEVICE_H

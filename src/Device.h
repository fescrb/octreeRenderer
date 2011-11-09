#ifndef _DEVICE_H
#define _DEVICE_H


class Device
{
    public:
						 Device();
        virtual 		~Device();

        virtual void	 printInfo() = 0;
    protected:
    private:
};

#endif // _DEVICE_H

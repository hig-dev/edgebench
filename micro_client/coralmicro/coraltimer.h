#if !defined(CORALTIMER_H)
#define CORALTIMER_H

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

class CoralTimer
{
public:
    CoralTimer()
    { 
	
    }

    CoralTimer(int ms)
    { 
		countdown_ms(ms);
    }

    static int getCurrentTime()
    {
        return xTaskGetTickCount() * portTICK_PERIOD_MS;
    }
    

    bool expired()
    {
		int now = getCurrentTime();
        int res = end_time - now;
        return res < 0;
    }
    

    void countdown_ms(int ms)  
    {
		int now = getCurrentTime();
		end_time = now + ms;
    }

    
    void countdown(int seconds)
    {
		int now = getCurrentTime();
        end_time = now + (seconds * 1000);
    }

    
    int left_ms()
    {
		int now = getCurrentTime();
		int res = end_time - now;
        return (res < 0) ? 0 : res;
    }
    
private:
	int end_time;
};

#endif
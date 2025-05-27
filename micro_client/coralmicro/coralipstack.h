#if !defined(CORALIPSTACK_H)
#define CORALIPSTACK_H

#include "coralmicro/libs/base/network.h"

class CoralIPStack
{
public:
    CoralIPStack(int sockfd)
    {
        sockfd_ = sockfd;
        open = false;
    }

    int connect(char *hostname, int port)
    {
        if (open)
        {
            disconnect();
        }

        sockfd_ = coralmicro::SocketClient(hostname, port);
        if (sockfd_ < 0)
        {
            printf("SocketClient failed to connect to %s:%d\r\n", hostname, port);
            return -1;
        }
        
        printf("SocketClient connected to %s:%d\r\n", hostname, port);
        return sockfd_;
    }

    /* returns the number of bytes read, which could be 0.
       -1 if there was an error on the socket
    */
    int read(unsigned char *buffer, int len, int timeout)
    {
        int interval = 10;  // all times are in milliseconds
		int total = 0, rc = -1;

		if (timeout < 30)
        {
            interval = 2;
        }
		
        while (coralmicro::SocketAvailable(sockfd_) < len && total < timeout)
		{
			vTaskDelay(interval / portTICK_PERIOD_MS);
			total += interval;
		}
		if (coralmicro::SocketAvailable(sockfd_) >= len)
		{
            auto ioStatus = coralmicro::ReadBytes(sockfd_, buffer, len);
            switch (ioStatus)
            {
                case coralmicro::IOStatus::kOk:
                    rc = len; // all bytes read
                    break;
                case coralmicro::IOStatus::kEof:
                    printf("SocketClient EOF\r\n");
                    rc = -1;
                    break;
                case coralmicro::IOStatus::kError:
                    printf("SocketClient Error\r\n");
                    rc = -1;
                    break;
                default:
                    rc = -1;
            }
        }
		return rc;
    }

    int write(unsigned char *buffer, int len, int timeout)
    {
        coralmicro::IOStatus status = coralmicro::WriteBytes( sockfd_, buffer, len );
        switch (status)
        {
            case coralmicro::IOStatus::kOk:
                return len; // all bytes written
            case coralmicro::IOStatus::kEof:
                printf("SocketClient EOF\r\n");
                return -1;
            case coralmicro::IOStatus::kError:
                printf("SocketClient Error\r\n");
                return -1;
            default:
                return -1;
        }
    }

    void disconnect()
    {
        open = false;
        coralmicro::SocketClose(sockfd_);
    }

private:
    bool open;
    int sockfd_;
};

#endif
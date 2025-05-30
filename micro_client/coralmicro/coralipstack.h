#if !defined(CORALIPSTACK_H)
#define CORALIPSTACK_H

#include "coralmicro/libs/base/network.h"

class CoralIPStack
{
public:
    CoralIPStack()
    {
        sockfd_ = -1;
        open = false;
    }

    int connect(const char *hostname, int port)
    {
        if (open)
        {
            disconnect();
        }

        sockfd_ = coralmicro::SocketClient(hostname, port);
        if (sockfd_ < 0)
        {
            return -1;
        }
        open = true;
        return sockfd_;
    }

    /* returns the number of bytes read, which could be 0 (timeout).
       -1 if there was an error on the socket
    */
    int read(unsigned char *buffer, int len, int timeout)
    {
        if (!open || sockfd_ < 0)
        {
            printf("SocketClient not connected\r\n");
            return -1; // not connected
        }
        coralmicro::IOStatus ioStatus = coralmicro::ReadBytes(sockfd_, buffer, len);
        switch (ioStatus)
        {
            case coralmicro::IOStatus::kOk:
                return len;
            case coralmicro::IOStatus::kEof:
                printf("SocketClient EOF\r\n");
                return -1; // EOF means no more data to read
            case coralmicro::IOStatus::kError:
                printf("SocketClient Error\r\n");
                return -1; // error reading from socket
            default:
                printf("SocketClient Unknown IOStatus\r\n");
                return -1; // unknown status
        }
    }

    int write(unsigned char *buffer, int len, int timeout)
    {
        if (!open || sockfd_ < 0)
        {
            printf("SocketClient not connected\r\n");
            return -1; // not connected
        }
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
        if (sockfd_ < 0)
        {
            printf("SocketClient already disconnected\r\n");
            return;
        }
        coralmicro::SocketClose(sockfd_);
    }

private:
    bool open;
    int sockfd_;
};

#endif
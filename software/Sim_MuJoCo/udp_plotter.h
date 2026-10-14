#pragma once
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

class UDPPlotter {
private:
    int sock_;
    struct sockaddr_in servaddr_;

public:
    UDPPlotter(const std::string& ip = "127.0.0.1", int port = 9870) {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        servaddr_.sin_family = AF_INET;
        servaddr_.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &servaddr_.sin_addr);
    }

    ~UDPPlotter() {
        close(sock_);
    }

    void send(const std::string& json_str) {
        sendto(sock_, json_str.c_str(), json_str.length(), 0, (const struct sockaddr *)&servaddr_, sizeof(servaddr_));
    }
};
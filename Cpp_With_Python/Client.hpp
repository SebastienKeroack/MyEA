#pragma once

class Client
{
    public:
        Client(void);
        
        ~Client(void);

        bool Initialized(void) const;

        bool Initialize(std::string const &script, std::string const &host);
        
        bool Open(void);
        
        bool Is_Connected(void);
        
        unsigned int CppCurrentTime(void);

        int Action(void);
        
        void Close(void);

        py::object main_module;
        py::object client;
};

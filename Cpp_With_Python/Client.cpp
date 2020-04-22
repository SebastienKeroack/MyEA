#include "pch.hpp"

#include <string>

#include "Client.hpp"

Client::Client(void) { }
        
Client::~Client(void) { }

void Write_To_File(std::string const text, std::string const path = "debug.txt")
{  
    std::ofstream file;

    file.open(path, std::ios_base::app);
    
    file << text << std::endl;

    file.close();
}

bool Client::Initialized(void) const
{
    return(Py_IsInitialized());
}

bool Client::Initialize(std::string const &script, std::string const &host)
{
    if(this->Initialized())
    {
        printf_s("Initialization can only be call once.\n");
        
        return(false);
    }
    
    try
    {
        Py_Initialize();

        np::initialize();
        
        this->main_module = py::import("__main__");
        
        py::object global(this->main_module.attr("__dict__"));

        // |STR| Prepare arguments. |STR|
        wchar_t **args = static_cast<wchar_t **>(PyMem_Malloc(3 * sizeof(wchar_t *)));
        
        args[0] = Py_DecodeLocale(boost::replace_all_copy(script, "\\", "\\\\").c_str(), NULL);
        args[1] = Py_DecodeLocale("--host", NULL);
        args[2] = Py_DecodeLocale(host.c_str(), NULL);
        
        PySys_SetArgv(3, args);
        // |END| Prepare arguments. |END|
        
        // Execute script.
        py::exec_file(script.c_str(),
                      global,
                      global);
        
        this->client = global["client"];
        
        return(true);
    }
    catch(py::error_already_set &)
    {
        Write_To_File("An error has been triggered from the `Initialize` function.");
        
        PyErr_Print();
        
        return(false);
    }
}
        
bool Client::Open(void)
{
    try
    {
        return(py::extract<bool>(this->client.attr("open")()));
    }
    catch(py::error_already_set &)
    {
        Write_To_File("An error has been triggered from the `Open` function.");
        
        PyErr_Print();
                
        return(false);
    }
}
        
bool Client::Is_Connected(void)
{
    try
    {
        return(py::extract<bool>(this->client.attr("is_connected")()));
    }
    catch(py::error_already_set &)
    {
        Write_To_File("An error has been triggered from the `Is_Connected` function.");
        
        PyErr_Print();
                
        return(false);
    }
}
        
unsigned int Client::CppCurrentTime(void)
{
    try
    {
        return(py::extract<unsigned int>(this->client.attr("CurrentTime")()));
    }
    catch(py::error_already_set &)
    {
        Write_To_File("An error has been triggered from the `CppCurrentTime` function.");
        
        PyErr_Print();
                
        return(0u);
    }
}

int Client::Action(void)
{
    try
    {
        return(py::extract<int>(this->client.attr("Action")()));
    }
    catch(py::error_already_set &)
    {
        Write_To_File("An error has been triggered from the `Action` function.");
        
        PyErr_Print();
                
        return(1);
    }
}
        
void Client::Close(void)
{
    try
    {
        this->client.attr("close")();
    }
    catch(py::error_already_set &)
    {
        Write_To_File("An error has been triggered from the `Close` function.");
        
        PyErr_Print();
    }
}

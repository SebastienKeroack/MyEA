#include "pch.hpp"

// This.
#include <API.hpp>
#include <Client/Client.hpp>

// Common_Library.
#include <Files/File.hpp>
#include <Strings/String.hpp>

class MyEA::RPC::Client g_Client;

DLL_API bool API__Cpp_Python_RPC__Initialize(void)
{
    if(g_Client.Initialized())
    {
        MyEA::File::fError("Initialization can only be call once per load." NEW_LINE
                           "Unload the `.dll` and retry.");
        
        return(false);
    }

    auto const results(Py_Try(&MyEA::RPC::Client::Initialize, std::ref(g_Client),
                              "C:\\Users\\sebas\\Documents\\MEGAsync\\MyEA\\Python\\run_client.py"));
    
    bool const &exception(!std::get<0>(results));

    if(exception || std::get<1>(results) == false)
    {
        if(exception) { MyEA::File::fError(PyErr_Parse().c_str()); }

        MyEA::File::fError("An error has been triggered from the `Initialize()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API bool API__Cpp_Python_RPC__Open(void)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(false);
    }

    auto const results(Py_Try(&MyEA::RPC::Client::Open, std::ref(g_Client)));
    
    bool const &exception(!std::get<0>(results));

    if(exception || std::get<1>(results) == false)
    {
        if(exception) { MyEA::File::fError(PyErr_Parse().c_str()); }

        MyEA::File::fError("An error has been triggered from the `Open()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API bool API__Cpp_Python_RPC__Close(void)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(false);
    }
    else if(Py_Try(&MyEA::RPC::Client::Close, std::ref(g_Client)) == false)
    {
        MyEA::File::fError("An error has been triggered from the `Close()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API size_t API__Cpp_Python_RPC__Sizeof_T(void)
{
    return(sizeof(T_));
}

DLL_API T_ API__Cpp_Python_RPC__Predict(T_ *const inputs)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }
    
    /*
    auto const results(Py_Try(&MyEA::RPC::Client::Predict, std::ref(g_Client),
                       inputs));
    
    bool const &exception(!std::get<0>(results));

    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str()); }

        MyEA::File::fError("An error has been triggered from the `Predict()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &model_metrics(std::get<1>(results));
    
    if(model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("Number array `model_metrics` is empty.");
        
        return(T_EMPTY());
    }

    return(py::extract<T_>(model_metrics[0]));
    */
    return(0);
}

DLL_API T_ API__Cpp_Python_RPC__Metric_Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }

    auto const results(Py_Try(&MyEA::RPC::Client::Model_Metrics, std::ref(g_Client)));
    
    bool const &exception(!std::get<0>(results));

    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An error has been triggered from the `Model_Metrics()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &model_metrics(std::get<1>(results));

    if(model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("Number array `model_metrics` is empty.");
        
        return(T_EMPTY());
    }

    return(py::extract<T_>(model_metrics[type_dataset][0][1]));
}

DLL_API T_ API__Cpp_Python_RPC__Metric_Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }
    
    auto const results(Py_Try(&MyEA::RPC::Client::Model_Metrics, std::ref(g_Client)));

    bool const &exception(!std::get<0>(results));

    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An error has been triggered from the `Model_Metrics()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &model_metrics(std::get<1>(results));
    
    if(model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("Number array `model_metrics` is empty.");
        
        return(T_EMPTY());
    }

    return(py::extract<T_>(model_metrics[type_dataset][1][1]));
}

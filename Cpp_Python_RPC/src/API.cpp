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

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Initialize, std::ref(g_Client),
                                  "C:\\Users\\sebas\\Documents\\MEGAsync\\MyEA\\Python\\run_client.py"));
    
    bool const &tmp_exception(!std::get<0>(tmp_results));

    if(tmp_exception || std::get<1>(tmp_results) == false)
    {
        if(tmp_exception) { MyEA::File::fError(PyErr_Parse().c_str()); }

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

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Open, std::ref(g_Client)));
    
    bool const &tmp_exception(!std::get<0>(tmp_results));

    if(tmp_exception || std::get<1>(tmp_results) == false)
    {
        if(tmp_exception) { MyEA::File::fError(PyErr_Parse().c_str()); }

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

DLL_API T_ API__Cpp_Python_RPC__Predict(T_ *const ptr_inputs_received)
{
    /*
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Predict, std::ref(g_Client),
                                  ptr_inputs_received));
    
    bool const &tmp_exception(!std::get<0>(tmp_results));

    if(tmp_exception)
    {
        if(tmp_exception) { MyEA::File::fError(PyErr_Parse().c_str()); }

        MyEA::File::fError("An error has been triggered from the `Predict()` function.");
        
        return(T_EMPTY());
    }

    return(std::get<1>(tmp_results));
    */
    return(0);
}

DLL_API T_ API__Cpp_Python_RPC__Metric_Loss(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }

    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Model_Metrics, std::ref(g_Client)));
    
    bool const &tmp_exception(!std::get<0>(tmp_results));

    if(tmp_exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An error has been triggered from the `Model_Metrics()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &tmp_ref_model_metrics(std::get<1>(tmp_results));

    if(tmp_ref_model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("Number array `tmp_ref_model_metrics` is empty.");
        
        return(T_EMPTY());
    }

    return(py::extract<T_>(tmp_ref_model_metrics[type_dataset_received][0][1]));
}

DLL_API T_ API__Cpp_Python_RPC__Metric_Accuracy(enum MyEA::Common::ENUM_TYPE_DATASET const type_dataset_received)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }
    
    auto const tmp_results(Py_Try(&MyEA::RPC::Client::Model_Metrics, std::ref(g_Client)));

    bool const &tmp_exception(!std::get<0>(tmp_results));

    if(tmp_exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An error has been triggered from the `Model_Metrics()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &tmp_ref_model_metrics(std::get<1>(tmp_results));
    
    if(tmp_ref_model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("Number array `tmp_ref_model_metrics` is empty.");
        
        return(T_EMPTY());
    }

    return(py::extract<T_>(tmp_ref_model_metrics[type_dataset_received][1][1]));
}

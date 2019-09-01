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
    if(g_Client.Initialized()) { return(true); }

    auto const results(Py_Try(&MyEA::RPC::Client::Initialize, std::ref(g_Client),
                              "C:\\Users\\sebas\\Documents\\MEGAsync\\MyEA\\Python\\run_client.py"));
    
    bool const &exception(!std::get<0>(results)),
               &error    (!std::get<1>(results));
    
    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An exception has been triggered from the `Initialize()` function.");
        
        return(false);
    }
    else if(error)
    {
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
    
    bool const &exception(!std::get<0>(results)),
               &error    (!std::get<1>(results));
    
    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An exception has been triggered from the `Open()` function.");
        
        return(false);
    }
    else if(error)
    {
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

DLL_API bool API__Cpp_Python_RPC__Concatenate_X(T_ const *const inputs, size_t const length)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(false);
    }
    
    py::tuple const shape(py::make_tuple(length));

    np::dtype const dtype(np::dtype::get_builtin<T_>());

    np::ndarray py_inputs(np::empty(shape, dtype));

    for(int i(0); i != length; ++i) { py_inputs[i] = inputs[i]; }

    auto const results(Py_Try(&MyEA::RPC::Client::Concatenate_X, std::ref(g_Client),
                              py_inputs));
    
    bool const &exception(!std::get<0>(results)),
               &error    (!std::get<1>(results));
    
    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An exception has been triggered from the `Concatenate_X()` function.");
        
        return(false);
    }
    else if(error)
    {
        MyEA::File::fError("An error has been triggered from the `Concatenate_X()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API bool API__Cpp_Python_RPC__Concatenate_Y(T_ const *const inputs, size_t const length)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(false);
    }
    
    py::tuple const shape(py::make_tuple(length));

    np::dtype const dtype(np::dtype::get_builtin<T_>());

    np::ndarray py_inputs(np::empty(shape, dtype));

    for(int i(0); i != length; ++i) { py_inputs[i] = inputs[i]; }

    auto const results(Py_Try(&MyEA::RPC::Client::Concatenate_Y, std::ref(g_Client),
                              py_inputs));
    
    bool const &exception(!std::get<0>(results)),
               &error    (!std::get<1>(results));
    
    if(exception)
    {
        MyEA::File::fError(PyErr_Parse().c_str());

        MyEA::File::fError("An exception has been triggered from the `Concatenate_Y()` function.");
        
        return(false);
    }
    else if(error)
    {
        MyEA::File::fError("An error has been triggered from the `Concatenate_Y()` function.");
        
        return(false);
    }

    return(true);
}

DLL_API size_t API__Cpp_Python_RPC__Sizeof_T(void)
{
    return(sizeof(T_));
}

DLL_API T_ API__Cpp_Python_RPC__Predict(unsigned int const past_action,
                                        T_ const *const inputs,
                                        size_t const length)
{
    if(g_Client.Initialized() == false)
    {
        MyEA::File::fError("Client is not initialized.");
        
        return(T_EMPTY());
    }
    
    np::dtype const dtype(np::dtype::get_builtin<T_>());

    auto Normalize_X([&dtype](T_ const *const inputs, size_t const length) -> np::ndarray
    {
        py::tuple const shape(py::make_tuple(length));

        np::ndarray py_inputs(np::empty(shape, dtype));

        for(int i(0); i != length; ++i) { py_inputs[i] = inputs[i]; }
        
        auto const results(Py_Try(&MyEA::RPC::Client::Normalize_X, std::ref(g_Client),
                                  py_inputs));
        
        bool const &exception(!std::get<0>(results));
        
        if(exception)
        {
            MyEA::File::fError(PyErr_Parse().c_str());

            MyEA::File::fError("An exception has been triggered from the `Normalize_X()` function.");
            
            return(np::from_object(py::object()));
        }
        
        return(std::get<1>(results));
    });
    
    auto Predict([](py::list const &inputs) -> T_
    {
        auto const results(Py_Try(&MyEA::RPC::Client::Predict, std::ref(g_Client),
                                  inputs));
        
        bool const &exception(!std::get<0>(results));
        
        if(exception)
        {
            MyEA::File::fError(PyErr_Parse().c_str());

            MyEA::File::fError("An exception has been triggered from the `Predict()` function.");
            
            return(T_EMPTY());
        }
        
        np::ndarray const &outputs(std::get<1>(results));
        
        if(outputs.get_nd() == 0)
        {
            MyEA::File::fError("NumPy array `outputs` is empty.");
            
            return(T_EMPTY());
        }
        
        if(outputs.get_nd() == 1) { return(py::extract<T_>(outputs[0]   )); }
        else                      { return(py::extract<T_>(outputs[0][0])); }
    });
    
    py::list list_of_inputs;

    // |STR| Financial features. |STR|
    np::ndarray const &X(Normalize_X(inputs, length));
    
    if(X.get_nd() == 0)
    {
        MyEA::File::fError("NumPy array `X` is empty.");
        
        return(T_EMPTY());
    }
    
    list_of_inputs.append(X);
    // |END| Financial features. |END|
    
    // |STR| Past actions. |STR|
    py::tuple const shape(py::make_tuple(1, 3));

    np::ndarray pA(np::zeros(shape, dtype));

    pA[0][past_action] = 1.0_T;

    list_of_inputs.append(pA);
    // |END| Past actions. |END|

    return(Predict(list_of_inputs));
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

        MyEA::File::fError("An exception has been triggered from the `Model_Metrics()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &model_metrics(std::get<1>(results));

    if(model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("NumPy array `model_metrics` is empty.");
        
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

        MyEA::File::fError("An exception has been triggered from the `Model_Metrics()` function.");
        
        return(T_EMPTY());
    }

    np::ndarray const &model_metrics(std::get<1>(results));
    
    if(model_metrics.get_nd() == 0)
    {
        MyEA::File::fError("NumPy array `model_metrics` is empty.");
        
        return(T_EMPTY());
    }

    return(py::extract<T_>(model_metrics[type_dataset][1][1]));
}

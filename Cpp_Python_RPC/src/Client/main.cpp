// cd /path/to/boost_1_70_0
// sudo gedit /usr/local/boost_1_70_0/libs/beast/tools/user-config.jam
// sudo bash bootstrap.sh --prefix=/usr/local/ --with-python-version=3.7
//  Add this line "using python : 3.7 : /usr/bin/python3 : /usr/include/python3.7m : /usr/lib ;" at the end of the file
// sudo b2 install -j8
// cd /path/to/project
// g++ -I/usr/include/python3.7 -c -fPIC hello_ext.cpp -o hello_ext.o
// g++ -shared -Wl,-soname,hello_ext.so -o hello_ext.so  hello_ext.o -lpython3.7m -lboost_python37

// Boost.
#include <boost/python.hpp>

// THIS.
#include <Client/Manager.hpp>

class Manager *g_Manager = nullptr;

void API__Allocate()
{
    if(g_Manager == nullptr)
    {
        g_Manager = new class Manager;
    }
}

void API__Deallocate()
{
    delete(g_Manager);
    g_Manager = nullptr;
}

BOOST_PYTHON_MODULE(RPC_Client)
{
    using namespace boost::python;

    def("Allocate",   API__Allocate  );
    def("Deallocate", API__Deallocate);
}
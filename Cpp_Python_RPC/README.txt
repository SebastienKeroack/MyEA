# Boost Python:
	# Download boost at https://www.boost.org/users/download/
	
	# Linux:
		BOOST_DIRECTORY = /usr/local/boost_1_70_0
		
		# Extract file to $(BOOST_DIRECTORY)
		# 	tar  x: This option tells tar to extract the files.
		# 	tar  v: The “v” stands for “verbose.” This option will list all of the files one by one in the archive.
		# 	tar  z: The z option is very important and tells the tar command to uncompress the file (gzip).
		# 	tar  f: This options tells tar that you are going to give it a file name to work with.
		# 	tar -C: Argument is used to specify the path to place the file.
		tar xvzf boost_1_70_0.tar.gz -C $(BOOST_DIRECTORY)

		# Go to $(BOOST_DIRECTORY)
		cd $(BOOST_DIRECTORY)
		
		# Specify the tools and libraries available to the build system.
		"using python : 3.7 : /usr/bin/python3 : /usr/include/python3.7m : /usr/lib ;" >> $(BOOST_DIRECTORY)/libs/beast/tools/user-config.jam
		
		# Boost's bootstrap setup.
		bash $(BOOST_DIRECTORY)/bootstrap.sh --prefix=/usr/local/ --with-python-version=3.7

		# Build.
		# 	b2 -j: Number of threads to use.
		bash b2 -j8

		# Install.
		# 	b2 -j: Number of threads to use.
		#???sudo b2 install -j8

		# To use custom module make a symbolic link to the `sites-packages` of the main python.
		ln -s /path/to/module /usr/lib/python3.7/site-packages

		# Test:
			# Go to project directory.
			cd /path/to/project

			# Build project using g++.
			# 	gcc    -I: Adds include directory of header files.
			# 	gcc    -c: Compile or assemble the source files, but do not link.
			# 	gcc -fPIC: Generates position independent code (PIC) for shared libraries.
			# 	gcc    -o: Writes the build output to an output file.
			g++ -I/usr/include/python3.7 -c -fPIC hello_ext.cpp -o hello_ext.o
			# 	gcc -shared: Generates shared object file for shared library.
			# 	gcc -Wl,xxx: Option for gcc passes a comma-separated list of tokens as a space-separated list of arguments to the linker.
			# 	gcc -soname: Specifies the required soname.
			# 	gcc      -l: Links with a library file.
			g++ -shared -Wl,-soname,hello_ext.so -o hello_ext.so  hello_ext.o -lpython3.7m -lboost_python37

	# Windows:
		BOOST_DIRECTORY  = "C:\\Program Files\\boost_1_70_0"
		PYTHON_DIRECTORY = C:\\Users\\sebas\\AppData\\Local\\Programs\\Python

		# Extract file to $(BOOST_DIRECTORY)
		..?
		
		# Go to $(BOOST_DIRECTORY)
		cd $(BOOST_DIRECTORY)
		
		# Open "Developer Command Prompt for VS 2019".
			# Boost's bootstrap setup.
			bootstrap.bat vc142

		# Open "Cygwin64 Terminal" or use a text editor.
			# Remove `using msvc ;` from the file.
			grep -Fvf "using msvc ;" project-config.jam
			
			# Append to the file...
			"# x64" >> project-config.jam
			"using msvc : 14.2
					   : "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Preview\\VC\\Tools\\MSVC\\14.23.27911\\bin\\Hostx64\\x64\\cl.exe"
					   : <address-model>64
					   ;" >> project-config.jam

			"# x64" >> project-config.jam
			"using python : 3.7
						 : $(PYTHON_DIRECTORY)\\Python37\\python.exe
						 : $(PYTHON_DIRECTORY)\\Python37\\include
						 : $(PYTHON_DIRECTORY)\\Python37\\libs
						 : <address-model>64
						 ;" >> project-config.jam

			"# x86" >> project-config.jam
			"using python : 3.7
						 : $(PYTHON_DIRECTORY)\\Python37-32\\python.exe
						 : $(PYTHON_DIRECTORY)\\Python37-32\\include
						 : $(PYTHON_DIRECTORY)\\Python37-32\\libs
						 : <address-model>32
						 ;" >> project-config.jam
		
		# Open "Developer Command Prompt for VS 2019".
			# Build.
			b2 --with-python --address-model=32,64 -j8 --build-type=complete
		
		# Create a symbolic link to the include path.
		ln -s include "C:\\Program Files\\include\\boost"
		ln -s $(PYTHON_DIRECTORY)\\Python37\\include "C:\\Program Files\\include\\python_x64"
		ln -s $(PYTHON_DIRECTORY)\\Python37-32\\include "C:\\Program Files\\include\\python_x86"
		
		# Create a symbolic link to the lib search path.
		ln -s stage\\lib "C:\\Program Files\\lib\\boost"
		ln -s $(PYTHON_DIRECTORY)\\Python37\\libs "C:\\Program Files\\lib\\python_x64"
		ln -s $(PYTHON_DIRECTORY)\\Python37-32\\libs "C:\\Program Files\\lib\\python_x86"

		# To use custom module make a symbolic link to the `sites-packages` of the main python.
		ln -s \\path\\to\\module $(PYTHON_DIRECTORY)\\Python37\\Lib\\site-packages
		ln -s \\path\\to\\module $(PYTHON_DIRECTORY)\\Python37-32\\Lib\\site-packages

# RPCLib:
	# Download rpclib at https://github.com/rpclib/rpclib
	git clone https://github.com/rpclib/rpclib.git
	
	# Linux:
		RPCLIB_DIRECTORY = /usr/local/rpclib
		
		# Move downloaded directory.
		mv rpclib $(RPCLIB_DIRECTORY)
		
		# Go to $(RPCLIB_DIRECTORY)
		cd $(RPCLIB_DIRECTORY)
		
		# Create `build` directory and go to.
		mkdir build && cd build
		
		# Pre build.
		cmake ..
		
		# Build debug.
		cmake --build .
		
		# Build.
		cmake --build . --config Release
		
		# Create a symbolic link to the lib path.
		ln -s Debug ..\\lib\\rpc\\Debug
		
		# Create a symbolic link to the lib path.
		ln -s Release ..\\lib\\rpc\\Release
		
		# Create a symbolic link to the include path.
		# ln -s: make symbolic links instead of hard links.
		ln -s include/rpc /usr/local/include/rpc
		
		# Create a symbolic link to the lib search path.
		# ln -s: make symbolic links instead of hard links.
		ln -s lib/rpc /usr/local/lib/rpc
		
	# Windows:
		RPCLIB_DIRECTORY = "C:\\Program Files\\rpclib"
		
		# Open "Developer Command Prompt for VS 2019".
			# Move downloaded directory.
			mv rpclib $(RPCLIB_DIRECTORY)
			
			# Go to $(RPCLIB_DIRECTORY)
			cd $(RPCLIB_DIRECTORY)
			
			# Create `build` directory and go to.
			mkdir build && cd build
			
			# Pre build.
			cmake ..
			
			# Build debug.
			cmake --build .
			
			# Build release.
			cmake --build . --config Release
			
			# Create a symbolic link to the lib path.
			ln -s Debug ..\\lib\\rpc\\Debug
			
			# Create a symbolic link to the lib path.
			ln -s Release ..\\lib\\rpc\\Release
			
			# Create a symbolic link to the include path.
			ln -s include\\rpc "C:\\Program Files\\include\\rpc"
			
			# Create a symbolic link to the lib search path.
			ln -s lib\\rpc "C:\\Program Files\\lib\\rpc"
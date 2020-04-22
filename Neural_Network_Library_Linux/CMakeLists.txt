CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(Neural_Network_Library_Linux VERSION 1.0.0 DESCRIPTION "No description.")

ADD_DEFINITIONS(-DCOMPILE_x64
                -DCOMPILE_DOUBLE
                -DCOMPILE_COUT
                -DCOMPILE_WINDOWS
                -DCOMPILE_UINPUT
                -DCOMPILE_BREAK_CODE
                -DCOMPILE_DEBUG
                -DCOMPILE_AUTODIFF
                -DAUTODIFF_TRACK_NON_FINITE_GRADIENTS
                -DAUTODIFF_RECORDING_PAUSABLE)

SET(CMAKE_CXX_STANDARD 17)

# |STR| Library. |STR|
INCLUDE_DIRECTORIES(include)

SET(SOURCE_FILES
    src/stdafx.cpp
    src/Neural_Network/Bayesian_Optimization.cpp
    src/Neural_Network/Dataset.cpp
    src/Neural_Network/Dataset__Cross_Validation.cpp
    src/Neural_Network/Dataset__Cross_Validation__Hyper_Optimization.cpp
    src/Neural_Network/Dataset__Mini_Batch.cpp
    src/Neural_Network/Dataset_Manager.cpp
    src/Neural_Network/Gaussian_Search.cpp
    src/Neural_Network/Grid_Search.cpp
    src/Neural_Network/Hyper_Optimization.cpp
    src/Neural_Network/Neural_Network.cpp
    src/Neural_Network/Neural_Network__Accuracy.cpp
    src/Neural_Network/Neural_Network__AdaBound.cpp
    src/Neural_Network/Neural_Network__Adam.cpp
    src/Neural_Network/Neural_Network__Allocate.cpp
    src/Neural_Network/Neural_Network__AMSBound.cpp
    src/Neural_Network/Neural_Network__AMSGrad.cpp
    src/Neural_Network/Neural_Network__Assign_Inputs.cpp
    src/Neural_Network/Neural_Network__Backward_Pass__Loop.cpp
    src/Neural_Network/Neural_Network__Backward_Pass__OpenMP.cpp
    src/Neural_Network/Neural_Network__Batch_Normalization.cpp
    src/Neural_Network/Neural_Network__Batch_Renormalization.cpp
    src/Neural_Network/Neural_Network__Clip_Gradient.cpp
    src/Neural_Network/Neural_Network__Compile.cpp
    src/Neural_Network/Neural_Network__Compute__Error__Loop.cpp
    src/Neural_Network/Neural_Network__Compute__Error__OpenMP.cpp
    src/Neural_Network/Neural_Network__Compute__Loss__Loop.cpp
    src/Neural_Network/Neural_Network__Compute__Loss__OpenMP.cpp
    src/Neural_Network/Neural_Network__Compute__R__Loop.cpp
    src/Neural_Network/Neural_Network__Compute__R__OpenMP.cpp
    src/Neural_Network/Neural_Network__Copy.cpp
    src/Neural_Network/Neural_Network__Deallocate.cpp
    src/Neural_Network/Neural_Network__Dropout.cpp
    src/Neural_Network/Neural_Network__Dropout__Bernoulli__Loop.cpp
    src/Neural_Network/Neural_Network__Dropout__Bernoulli__OpenMP.cpp
    src/Neural_Network/Neural_Network__Dropout__Zoneout__Loop.cpp
    src/Neural_Network/Neural_Network__Dropout__Zoneout__OpenMP.cpp
    src/Neural_Network/Neural_Network__Forward_Pass__Loop.cpp
    src/Neural_Network/Neural_Network__Forward_Pass__OpenMP.cpp
    src/Neural_Network/Neural_Network__Gradient_Descent.cpp
    src/Neural_Network/Neural_Network__Gradient_Descent_Momentum.cpp
    src/Neural_Network/Neural_Network__Indexing.cpp
    src/Neural_Network/Neural_Network__Initialization__Glorot.cpp
    src/Neural_Network/Neural_Network__Initialization__Identity.cpp
    src/Neural_Network/Neural_Network__Initialization__LSUV.cpp
    src/Neural_Network/Neural_Network__Initialization__Orthogonal.cpp
    src/Neural_Network/Neural_Network__Initialization__Uniform.cpp
    src/Neural_Network/Neural_Network__iRPROP_minus.cpp
    src/Neural_Network/Neural_Network__iRPROP_plus.cpp
    src/Neural_Network/Neural_Network__K_Sparse.cpp
    src/Neural_Network/Neural_Network__Layer_Normalization.cpp
    src/Neural_Network/Neural_Network__Load.cpp
    src/Neural_Network/Neural_Network__Loss.cpp
    src/Neural_Network/Neural_Network__Merge.cpp
    src/Neural_Network/Neural_Network__Nesterov_Accelerated_Gradient.cpp
    src/Neural_Network/Neural_Network__NosAdam.cpp
    src/Neural_Network/Neural_Network__OpenMP.cpp
    src/Neural_Network/Neural_Network__Ordered.cpp
    src/Neural_Network/Neural_Network__Orthogonal.cpp
    src/Neural_Network/Neural_Network__Plot__Gradient.cpp
    src/Neural_Network/Neural_Network__Prepare_Connections.cpp
    src/Neural_Network/Neural_Network__Randomize.cpp
    src/Neural_Network/Neural_Network__Reallocate_Batch.cpp
    src/Neural_Network/Neural_Network__Reallocate_Parameters.cpp
    src/Neural_Network/Neural_Network__Reallocate_Threads.cpp
    src/Neural_Network/Neural_Network__Recurrent_Backward_Pass__Loop.cpp
    src/Neural_Network/Neural_Network__Recurrent_Backward_Pass__OpenMP.cpp
    src/Neural_Network/Neural_Network__Recurrent_Compute__Error__Loop.cpp
    src/Neural_Network/Neural_Network__Recurrent_Compute__Error__OpenMP.cpp
    src/Neural_Network/Neural_Network__Recurrent_Compute__Loss__Loop.cpp
    src/Neural_Network/Neural_Network__Recurrent_Compute__Loss__OpenMP.cpp
    src/Neural_Network/Neural_Network__Recurrent_Compute__R__Loop.cpp
    src/Neural_Network/Neural_Network__Recurrent_Compute__R__OpenMP.cpp
    src/Neural_Network/Neural_Network__Recurrent_Forward_Pass__Loop.cpp
    src/Neural_Network/Neural_Network__Recurrent_Forward_Pass__OpenMP.cpp
    src/Neural_Network/Neural_Network__Recurrent_Update_Derivative_Weight__Loop.cpp
    src/Neural_Network/Neural_Network__Recurrent_Update_Derivative_Weight__OpenMP.cpp
    src/Neural_Network/Neural_Network__Regularization__Constraint_Recurrent_Weight.cpp
    src/Neural_Network/Neural_Network__Regularization__L1.cpp
    src/Neural_Network/Neural_Network__Regularization__L2.cpp
    src/Neural_Network/Neural_Network__Regularization__Max_Norm_Constraints.cpp
    src/Neural_Network/Neural_Network__Regularization__SRIP.cpp
    src/Neural_Network/Neural_Network__Regularization__Weight_Decay.cpp
    src/Neural_Network/Neural_Network__Save.cpp
    src/Neural_Network/Neural_Network__Tied__Normalization.cpp
    src/Neural_Network/Neural_Network__Tied__Parameter.cpp
    src/Neural_Network/Neural_Network__Tied__Weight.cpp
    src/Neural_Network/Neural_Network__Tied_Parameter.cpp
    src/Neural_Network/Neural_Network__Tied_Weight.cpp
    src/Neural_Network/Neural_Network__Transfer_Learning.cpp
    src/Neural_Network/Neural_Network__Update.cpp
    src/Neural_Network/Neural_Network__Update_Derivative_Weight__Loop.cpp
    src/Neural_Network/Neural_Network__Update_Derivative_Weight__OpenMP.cpp
    src/Neural_Network/Neural_Network__User_Controls.cpp
    src/Neural_Network/Neural_Network__Xavier.cpp
    src/Neural_Network/Neural_Network_Manager.cpp
    src/Neural_Network/Train.cpp)

ADD_LIBRARY(Neural_Network_Library_Linux SHARED ${SOURCE_FILES})
# |END| Library. |END|

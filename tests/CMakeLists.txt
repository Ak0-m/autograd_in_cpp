
add_executable(test_value test_value.cpp)
target_link_libraries(test_value PRIVATE ag)
add_test(NAME ValueTests COMMAND $<TARGET_FILE:test_value>)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE ag)
add_test(NAME TensorTests COMMAND $<TARGET_FILE:test_tensor>)

add_executable(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose PRIVATE ag)
add_test(NAME TransposeTests COMMAND $<TARGET_FILE:test_transpose>)

add_executable(test_matmul test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE ag)
add_test(NAME MatmulTests COMMAND $<TARGET_FILE:test_matmul>)
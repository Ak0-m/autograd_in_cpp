# ---- Test executable for Value class ----
add_executable(test_value
    test_value.cpp
)

# ---- Link against your main library ----
target_link_libraries(test_value
    PRIVATE
        ag          # <--- FIXED: matches the library name in root CMake
)

# ---- Register the test with CTest ----
add_test(
    NAME ValueTests
    COMMAND test_value
)

# ---- (Future) Tensor tests ----
# Uncomment these when you create test_tensor.cpp in Week 2
# add_executable(test_tensor test_tensor.cpp)
# target_link_libraries(test_tensor PRIVATE ag)
# add_test(NAME TensorTests COMMAND test_tensor)
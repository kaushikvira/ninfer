# CPU-only: the serial tail's policy math is host code with no CUDA dependency, so the
# contract is verifiable on any box (the device op it mirrors is covered separately in
# tests/ops/dspine). The oracle is the committed fixture from make_fixture.py.
ninfer_add_test(ninfer_dspine_serial_tail_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_serial_tail.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp")
set_tests_properties(ninfer_dspine_serial_tail_test PROPERTIES TIMEOUT 120)

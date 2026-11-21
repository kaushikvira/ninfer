# DSpine serial tail: device op vs the committed reference fixture and the host core
# specification (models/common/dspine/dspine_serial_tail.cpp). Skips without a CUDA device.
ninfer_add_op_test(ninfer_dspine_serial_tail_op_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/serial_tail.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_ops)

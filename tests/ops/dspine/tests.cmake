# DSpine serial tail: device op vs the committed reference fixture and the host core
# specification (models/common/dspine/dspine_serial_tail.cpp). Skips without a CUDA device.
ninfer_add_op_test(ninfer_dspine_serial_tail_op_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/serial_tail.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_ops)

# The DSpine proposal path's nine linear geometries against the compiled shape tables. The linear
# families resolve a geometry on the host, before any launch, so this needs no device and no
# artifact: it is the gate the real-artifact probe measures, with the drafter's numbers folded in.
# Fails when a proposal step's geometry is not instantiated, and covers the prod tower's geometries
# so a shape-table regression fails here too.
ninfer_add_test(ninfer_dspine_shapes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/shapes.cpp"
  LIBRARIES ninfer_ops)

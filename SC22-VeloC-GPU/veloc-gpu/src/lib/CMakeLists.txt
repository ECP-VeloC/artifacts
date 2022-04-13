cuda_add_library (veloc-client SHARED
  veloc.cpp
  mem_region.cpp
  ${PROJECT_SOURCE_DIR}/src/backend/work_queue.cpp
  memory_cache.cpp
  client.cpp
)

# Link the executable to the necessary libraries.
# target_link_libraries (veloc-client ${CUDA_NVTX_LIBRARY})
target_link_libraries (veloc-client veloc-modules ${COMM_QUEUE_LIBRARIES})

# Install libraries
install (TARGETS veloc-client
  EXPORT veloc-clientConfig
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

target_include_directories(veloc-client PUBLIC 
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
)

install(EXPORT veloc-clientConfig DESTINATION lib64/cmake/${PROJECT_NAME})

install (FILES
  ${PROJECT_SOURCE_DIR}/include/veloc.h
  ${PROJECT_SOURCE_DIR}/include/veloc.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install (DIRECTORY
  ${PROJECT_SOURCE_DIR}/include/veloc
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

add_library (veloc-modules SHARED
  module_manager.cpp
  client_watchdog.cpp transfer_module.cpp chksum_module.cpp versioning_module.cpp # simple modules
  client_aggregator.cpp ec_module.cpp # aggregation modules
  ${PROJECT_SOURCE_DIR}/src/common/config.cpp
  ${PROJECT_SOURCE_DIR}/src/common/file_util.cpp
)

target_include_directories(veloc-modules PUBLIC
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
)

target_link_libraries(veloc-modules ${MPI_CXX_LIBRARIES} ${ER_LIBRARIES} ${AXL_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARY})

# Install libraries
install (TARGETS veloc-modules
  EXPORT veloc-modulesConfig
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT veloc-modulesConfig DESTINATION lib64/cmake/${PROJECT_NAME})
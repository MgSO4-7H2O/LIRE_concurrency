cmake_minimum_required(VERSION 3.18)
project(faiss_baseline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep Faiss internal sources untouched; build and link the library only.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_RAFT OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL generic CACHE STRING "" FORCE)

add_subdirectory(third_party/faiss EXCLUDE_FROM_ALL)

find_package(OpenMP REQUIRED)

set(FAISS_BASELINE_TARGETS
    faiss_ivfflat_lire_run_eval
    faiss_lire_concurrent_eval
)

foreach(tgt IN LISTS FAISS_BASELINE_TARGETS)
    add_executable(${tgt}
        src/${tgt}.cpp
    )

    if(tgt STREQUAL "faiss_ivfflat_lire_run_eval" OR
       tgt STREQUAL "faiss_lire_concurrent_eval")
        target_sources(${tgt} PRIVATE
            src/faiss_ivfflat_lire.cpp
        )
    endif()

    target_include_directories(${tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/faiss
    )

    if(TARGET faiss)
        target_link_libraries(${tgt} PRIVATE
            faiss
            OpenMP::OpenMP_CXX
        )
    elseif(TARGET faiss_avx2)
        target_link_libraries(${tgt} PRIVATE
            faiss_avx2
            OpenMP::OpenMP_CXX
        )
    else()
        message(FATAL_ERROR
            "Could not find a Faiss target (faiss or faiss_avx2)."
        )
    endif()
endforeach()

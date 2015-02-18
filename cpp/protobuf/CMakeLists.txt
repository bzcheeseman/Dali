find_package(Protobuf REQUIRED)

# COMPILING PROTOBUFS
set(PROTOBUF_DIR "${PROJECT_SOURCE_DIR}/protobuf")
file(GLOB ProtoFiles "${PROTOBUF_DIR}/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

# THIS A HACK TO GET PROTOBUFS TO BE INCLUDABLE...
add_library(proto
        ${ProtoSources}
        ${ProtoHeaders}
)

foreach(header ${ProtoHeaders})
    ADD_CUSTOM_COMMAND(TARGET proto
        COMMAND cmake -E copy ${header}
        ${PROTOBUF_DIR}
        DEPENDS ${header})
endforeach(header)


target_link_libraries(proto ${PROTOBUF_LIBRARIES})

set(upstream "${TDIB_SOURCE_DIR}/core/solverTD.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${upstream}")
file(READ "${upstream}" header)

function(edit_header from to)
    string(FIND "${header}" "${from}" location)
    if(location EQUAL -1)
        message(FATAL_ERROR "Unexpected TDIB-CCD header. Use the revision in benchmarks/chen/README.md")
    endif()
    string(REPLACE "${from}" "${to}" header "${header}")
    set(header "${header}" PARENT_SCOPE)
endfunction()

edit_header("initTimeIntv = Array2d(0,DeltaT)) {"
            "initTimeIntv = Array2d(0,DeltaT), double margin = 0) {")
edit_header("std::vector<Line> lines2){"
            "std::vector<Line> lines2, double projected_margin){")
edit_header("l.b += 1e-12 * abs(l.b);"
            "l.b += projected_margin + 1e-12 * abs(l.b);")
edit_header("AxisCheck(ptLines1, ptLines2);" "AxisCheck(ptLines1, ptLines2, margin * axis.norm());")
edit_header("AxisCheck(ptLines2, ptLines1);" "AxisCheck(ptLines2, ptLines1, margin * axis.norm());")
edit_header("const double upperTime = DeltaT) {"
            "const double upperTime = DeltaT, double margin = 0, std::size_t* nodes = nullptr,
                            std::size_t node_limit = std::numeric_limits<std::size_t>::max()) {")
edit_header("colTime, bb, initTimeIntv))" "colTime, bb, initTimeIntv, margin))")
edit_header("colTime, bb, cur.tIntv))" "colTime, bb, cur.tIntv, margin))")
edit_header("while (!heap.empty()) {" "while (!heap.empty()) {
            if (nodes && *nodes >= node_limit) return -2;
            if (nodes) ++*nodes;")
edit_header("exit(-1);" "throw std::runtime_error(\"TDIB boundary intersection failed\");")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/solverTD_margin.h" CONTENT "${header}")

package com.huskerdev.gpkt.utils


// Example output: __kernel void main(float*a, int b){
fun appendCFunctionHeader(
    buffer: StringBuilder,
    modifiers: List<String>,
    type: String,
    name: String,
    args: List<String>
){
    modifiers.joinTo(buffer, separator = " ")
    if(modifiers.isNotEmpty())
        buffer.append(" ")

    buffer.append(type).append(" ").append(name).append("(")
    args.joinTo(buffer, separator = ",")
    buffer.append("){")
}

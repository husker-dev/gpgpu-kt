package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.FunctionCallExpression
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.CProgramPrinter


class OpenCLProgram(
    override val context: OpenCLContext,
    ast: GPScope
): GPProgram(ast) {
    private val cl = context.opencl
    val program: CLProgram
    val kernel: CLKernel
    val maxGroupSize: Long

    override var released = false

    init {
        val prog = OpenCLProgramPrinter(ast, buffers, locals).stringify()

        program = cl.compileProgram(context.device.peer, context.peer, prog)
        kernel = cl.createKernel(program, "__m")
        maxGroupSize = clGetKernelWorkGroupInfo(kernel, context.device.peer, CL_KERNEL_WORK_GROUP_SIZE)[0]
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        buffers.forEachIndexed { i, field ->
            when(val value = map[field.name]!!){
                is Float -> cl.setArgument1f(kernel, i, value)
                is Int -> cl.setArgument1i(kernel, i, value)
                is Byte -> cl.setArgument1b(kernel, i, value)
                is Boolean -> cl.setArgument1b(kernel, i, if(value) 1 else 0)
                is OpenCLMemoryPointer<*> -> cl.setArgument(kernel, i, value.mem)
                else -> throw UnsupportedOperationException()
            }
        }
        cl.setArgument1i(kernel, buffers.size, instances) // Set count variable
        cl.setArgument1i(kernel, buffers.size+1, indexOffset) // Set offset variable
        cl.executeKernel(context.commandQueue, kernel, maxGroupSize, instances.toLong())
    }

    override fun release() {
        if(released) return
        context.releaseProgram(this)
        released = true
    }
}

private class OpenCLProgramPrinter(
    ast: GPScope,
    buffers: List<GPField>,
    locals: List<GPField>
): CProgramPrinter(ast, buffers, locals) {

    override fun stringifyModifiersInStruct(field: GPField) =
        stringifyModifiersInLocal(field)

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is GPField && obj.isConstant) "__constant"
        else if(obj is GPFunction && obj.returnType.isDynamicArray) "__global" else ""

    override fun stringifyModifiersInLocal(field: GPField) =
        if(field.type.isDynamicArray) "__global" else ""

    override fun stringifyModifiersInArg(field: GPField) =
        stringifyModifiersInLocal(field)

    override fun stringifyModifiersInLocalsStruct() = ""

    override fun stringifyMainFunctionDefinition(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("__kernel ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.toString(),
            name = "__m",
            args = buffers.map {
                if (it.type.isArray) "__global ${toCType(header, it.type)}* restrict __v${it.obfName}"
                else "${toCType(header, it.type)} __v${it.obfName}"
            } + listOf("int __c", "int __o")
        )
    }

    override fun stringifyMainFunctionBody(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("int ${function.arguments[0].obfName}=get_global_id(0)+__o;")
        buffer.append("if(${function.arguments[0].obfName}>=__c+__o)return;")
    }

    override fun convertArrayName(name: String, size: Int) =
        if(size == -1) "* restrict $name"
        else super.convertArrayName(name, size)

    override fun convertPredefinedFieldName(field: GPField) = when(field.name){
        "PI" -> "M_PI"
        "E" -> "M_E"
        "NaN" -> "NAN"
        "FLOAT_MAX" -> "FLT_MAX"
        "FLOAT_MIN" -> "FLT_MIN"
        "INT_MAX" -> "INT_MAX"
        "INT_MIN" -> "INT_MIN"
        else -> field.obfName
    }

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) = when(functionExpression.function.name){
        "abs" -> "fabs"
        "isNaN" -> "isnan"
        else -> functionExpression.function.obfName
    }
}
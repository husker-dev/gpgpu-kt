package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.CProgramPrinter


class CudaProgram(
    override val context: CudaContext,
    ast: GPScope
): GPProgram(ast){
    val module: CUmodule
    private val function: CUfunction
    override var released = false

    init {
        val prog = CudaProgramPrinter(ast, buffers, locals).stringify()

        module = Cuda.compileToModule(context.peer, prog)
        function = Cuda.getFunctionPointer(context.peer, module, "__m")
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val arrays = buffers.map { field ->
            when(val value = map[field.name]!!){
                is Float, is Int, is Byte, is Boolean -> value
                is CudaMemoryPointer<*> -> value.ptr
                else -> throw UnsupportedOperationException()
            }
        }.toTypedArray()
        Cuda.launch(
            context.device.peer, context.peer,
            function, instances,
            instances, indexOffset, *arrays)
    }

    override fun release() {
        if(released) return
        context.releaseProgram(this)
        released = true
    }
}

private class CudaProgramPrinter(
    ast: GPScope,
    buffers: List<GPField>,
    locals: List<GPField>
): CProgramPrinter(ast, buffers, locals,
    useExternC = true,
    useArrayStructCast = false
) {
    override fun stringifyMainFunctionDefinition(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("__global__ ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.toString(),
            name = "__m",
            args = listOf("int __c", "int __o") + buffers.map {
                if (it.type.isArray) "${toCType(header, it.type)}* __restrict__ __v${it.obfName}"
                else "${toCType(header, it.type)} __v${it.obfName}"
            }
        )
    }

    override fun stringifyMainFunctionBody(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("const int ${function.arguments[0].obfName}=blockIdx.x*blockDim.x+threadIdx.x+__o;")
        buffer.append("if(${function.arguments[0].obfName}>=__c+__o)return;")
    }

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is GPField && obj.isConstant) "__constant__"
        else "__device__"

    override fun stringifyModifiersInStruct(field: GPField) = ""
    override fun stringifyModifiersInLocal(field: GPField) = ""
    override fun stringifyModifiersInArg(field: GPField) = ""
    override fun stringifyModifiersInLocalsStruct() = ""

    override fun convertArrayName(name: String, size: Int) =
        if(size == -1) "* __restrict__ $name"
        else super.convertArrayName(name, size)

    override fun convertPredefinedFieldName(field: GPField) = when(field.name){
        "PI" -> "3.141592653589793"
        "E" -> "2.718281828459045"
        "NaN" -> "__int_as_float(0x7fffffff)"
        "FLOAT_MAX" -> "3.402823e+38"
        "FLOAT_MIN" -> "1.175494e-38"
        "INT_MAX" -> "2147483647"
        "INT_MIN" -> "−2147483648"
        else -> field.obfName
    }

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) = when(functionExpression.function.name){
        "isNaN" -> "isnan"
        else -> functionExpression.function.obfName
    }
}
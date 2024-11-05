package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.utils.CProgramPrinter


class CudaProgram(
    private val context: CudaContext,
    ast: ScopeStatement
): GPProgram(ast){
    private val module: CUmodule
    private val function: CUfunction

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

    override fun dealloc() = Unit
}

private class CudaProgramPrinter(
    ast: ScopeStatement,
    buffers: List<GPField>,
    locals: List<GPField>
): CProgramPrinter(ast, buffers, locals,
    useExternC = true,
    useLocalStruct = false,
    useArrayStructCast = false
) {
    override fun stringifyMainFunctionDefinition(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("__global__ ")
        com.huskerdev.gpkt.utils.appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.toString(),
            name = "__m",
            args = listOf("int __c", "int __o") + buffers.map {
                if (it.type.isArray) "${toCType(header, it.type)}*__v${it.obfName}"
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
        buffers.joinTo(buffer, separator = ""){
            "${it.obfName}=__v${it.obfName};"
        }
    }

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is GPField && obj.isConstant) "__constant__"
        else "__device__"

    override fun stringifyModifiersInStruct(field: GPField) = ""
    override fun stringifyModifiersInLocal(field: GPField) = ""
    override fun stringifyModifiersInArg(field: GPField) = ""

    override fun convertPredefinedFieldName(field: GPField) = when(field.name){
        "PI" -> "3.141592653589793"
        "E" -> "2.718281828459045"
        "NaN" -> "__int_as_float(0x7fffffff)"
        else -> field.obfName
    }

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) = when(functionExpression.function.name){
        "isNaN" -> "isnan"
        else -> functionExpression.function.obfName
    }
}
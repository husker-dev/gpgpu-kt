package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.utils.SimpleCProgram
import com.huskerdev.gpkt.utils.appendCFunctionDefinition


class CudaProgram(
    private val context: CudaContext,
    ast: ScopeStatement
): SimpleCProgram(ast, false) {
    private val cuda = context.cuda

    private val module: CUmodule
    private val function: CUfunction

    init {
        val buffer = StringBuilder()
        buffer.append("extern \"C\"{")
        stringifyScopeStatement(buffer, ast, false)
        buffer.append("}")

        module = cuda.compileToModule(context.peer, buffer.toString())
        function = cuda.getFunctionPointer(context.peer, module, "__m")
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val arrays = buffers.map { field ->
            when(val value = map[field.name]!!){
                is Float, is Int, is Byte -> value
                is CudaMemoryPointer<*> -> value.ptr
                else -> throw UnsupportedOperationException()
            }
        }.toTypedArray()
        cuda.launch(
            context.device.peer, context.peer,
            function, instances,
            instances, indexOffset, *arrays)
    }

    override fun dealloc() = Unit

    override fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: GPFunction) {
        buffer.append("__global__ ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.toString(),
            name = "__m",
            args = listOf("int __c", "int __o") + buffers.map {
                if(it.type.isArray) "${toCType(it.type)}*__v${it.name}"
                else "${toCType(it.type)} __v${it.name}"
            }
        )
    }

    override fun stringifyMainFunctionBody(buffer: StringBuilder, function: GPFunction) {
        buffer.append("const int ${function.arguments[0].name}=blockIdx.x*blockDim.x+threadIdx.x+__o;")
        buffer.append("if(${function.arguments[0].name}>__c+__o)return;")
        buffers.joinTo(buffer, separator = ""){
            "${it.name}=__v${it.name};"
        }
    }

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is Field && obj.isConstant) "__constant__"
        else "__device__"

    override fun stringifyModifiersInStruct(field: Field) = ""
    override fun stringifyModifiersInLocal(field: Field) = ""
    override fun stringifyModifiersInArg(field: Field) = ""

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("3.141592653589793")
            "E" -> buffer.append("2.718281828459045")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }
}
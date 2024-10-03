package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.FieldExpression
import com.huskerdev.gpkt.ast.FunctionStatement
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import com.huskerdev.gpkt.utils.useStack


class CudaProgram(
    private val cuda: Cuda,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val module: Long
    private val function: Long

    init {
        val buffer = StringBuilder()
        buffer.append("extern \"C\"{")
        stringifyScopeStatement(buffer, ast, false)
        buffer.append("}")

        module = cuda.compileToModule(buffer.toString())
        function = cuda.getFunctionPointer(module, "__m")
    }

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) = useStack {
        val map = hashMapOf(*mapping)

        val values = mallocPointer(mapping.size + 2)
        values.put(0, ints(instances))
        values.put(1, ints(indexOffset))

        buffers.forEachIndexed { i, field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            val o = i+2
            when(value){
                is Float -> values.put(o, floats(value))
                is Double -> values.put(o, doubles(value))
                is Int -> values.put(o, ints(value))
                is Byte -> values.put(o, bytes(value))
                is CudaMemoryPointer<*> -> values.put(o, value.ptr)
                else -> throw UnsupportedOperationException()
            }
        }
        cuda.launch(function, instances, values)
        Unit
    }

    override fun dealloc() = Unit

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("3.141592653589793")
            "E" -> buffer.append("2.718281828459045")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        if(function.name == "main") {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__global__"),
                type = function.returnType.text,
                name = "__m",
                args = listOf("int __c", "int __o") + buffers.map(::transformKernelArg)
            )
            buffer.append("{")
            buffer.append("const int ${function.arguments[0].name}=blockIdx.x*blockDim.x+threadIdx.x+__o;")
            buffer.append("if(${function.arguments[0].name}>__c+__o)return;")
            buffers.forEach {
                buffer.append("${it.name}=__v_${it.name};")
            }
            stringifyScopeStatement(buffer, statement.function.body, false)
            buffer.append("}")
        }else {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__device__") + function.modifiers.map { it.text },
                type = function.returnType.text,
                name = function.name,
                args = function.arguments.map(::convertToFuncArg)
            )
            stringifyScopeStatement(buffer, statement.function.body, true)
        }
    }

    private fun transformKernelArg(field: Field): String{
        return if(field.type.isArray)
            "${toCType(field.type)}*__v_${field.name}"
        else
            "${toCType(field.type)} __v_${field.name}"
    }


    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> "__device__"
        Modifiers.CONST -> "__constant__"
        Modifiers.READONLY -> ""
    }
}
package com.huskerdev.gpkt.engines.metal

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.objcPtr
import kotlinx.cinterop.objc_release
import platform.Metal.MTLComputeCommandEncoderProtocol
import platform.Metal.MTLComputePipelineStateProtocol
import platform.Metal.MTLFunctionProtocol
import platform.Metal.MTLLibraryProtocol

class MetalProgram(
    private val metal: Metal,
    ast: ScopeStatement
): SimpleCProgram(ast) {

    private val library: MTLLibraryProtocol
    private val function: MTLFunctionProtocol
    private val pipeline: MTLComputePipelineStateProtocol
    private val commandEncoder: MTLComputeCommandEncoderProtocol

    init {
        val buffer = StringBuilder()

        buffer.append("typedef struct{")
        buffers.joinTo(buffer, separator = ";", postfix = ";") { convertToFuncArg(it) }
        buffer.append("}__in;")

        stringifyScopeStatement(buffer, ast, false)

        library = metal.createLibrary(buffer.toString())
        function = metal.getFunction(library, "_m")!!
        pipeline = metal.createPipeline(metal.device, function)
        commandEncoder = metal.createCommandEncoder(pipeline)
    }

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        buffers.forEachIndexed { i, field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            when(value){
                is Float -> metal.setFloatAt(commandEncoder, value, i)
                is Int -> metal.setIntAt(commandEncoder, value, i)
                is Byte -> metal.setByteAt(commandEncoder, value, i)
                is MetalMemoryPointer<*> -> metal.setBufferAt(commandEncoder, value.buffer, i)
                else -> throw UnsupportedOperationException()
            }
        }
        metal.setIntAt(commandEncoder, indexOffset, buffers.size)
        metal.execute(commandEncoder, instances)
    }

    @OptIn(ExperimentalForeignApi::class)
    override fun dealloc() {
        objc_release(library.objcPtr())
        objc_release(function.objcPtr())
        objc_release(pipeline.objcPtr())
        objc_release(commandEncoder.objcPtr())
    }

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder) {
        val function = statement.function
        if(function.name == "main"){
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("kernel"),
                type = "void",
                name = "_m",
                args = buffers.map(::transformKernelArg) +
                        listOf("device int*__o", "uint i [[thread_position_in_grid]]")
            )
            buffer.append("{")

            // Struct with inputs
            buffer.append("__in __v={")
            buffers.forEachIndexed { index, field ->
                buffer.append("__v").append(field.name)
                if(!field.type.isArray)
                    buffer.append("[0]")
                if(index != buffers.lastIndex)
                    buffer.append(",")
            }
            buffer.append("};")

            // Body
            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        } else {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = function.modifiers.map { it.text },
                type = toCType(function.returnType),
                name = function.name,
                args = listOf("__in __v") + function.arguments.map(::convertToFuncArg)
            )
            stringifyStatement(buffer, statement.function.body)
        }
    }

    private fun transformKernelArg(field: Field): String{
        return if(field.type.isArray)
            "${toCType(field.type)}*__v${field.name}"
        else
            "device ${toCType(field.type)}*__v${field.name}"
    }

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        if(expression.field in buffers)
            buffer.append("__v.")
        when(expression.field.name){
            "PI" -> buffer.append("M_PI")
            "E" -> buffer.append("M_E")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }

    override fun stringifyArrayAccessExpression(
        buffer: StringBuilder,
        expression: ArrayAccessExpression
    ) {
        if(Modifiers.EXTERNAL in expression.array.modifiers)
            buffer.append("__v.")
        super.stringifyArrayAccessExpression(buffer, expression)
    }

    override fun stringifyFunctionCallExpression(
        buffer: StringBuilder,
        expression: FunctionCallExpression
    ) {
        buffer.append(expression.function.name)
        buffer.append("(")
        buffer.append("__v")
        if(expression.arguments.isNotEmpty())
            buffer.append(",")

        expression.arguments.forEachIndexed { i, arg ->
            stringifyExpression(buffer, arg)
            if(i != expression.arguments.lastIndex)
                buffer.append(",")
        }
        buffer.append(")")
    }

    override fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder) {
        if(Modifiers.EXTERNAL !in fieldStatement.fields[0].modifiers)
            super.stringifyFieldStatement(fieldStatement, buffer)
    }

    override fun toCType(type: Type) = when(type){
        Type.FLOAT_ARRAY -> "device float"
        Type.INT_ARRAY -> "device int"
        Type.BYTE_ARRAY -> "device char"
        Type.BOOLEAN_ARRAY -> "device bool"
        else -> super.toCType(type)
    }

    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> ""
        Modifiers.CONST -> "constant"
        Modifiers.READONLY -> ""
    }
}
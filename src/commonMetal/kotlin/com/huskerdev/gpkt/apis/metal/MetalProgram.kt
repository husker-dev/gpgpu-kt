package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.utils.SimpleCProgram
import com.huskerdev.gpkt.utils.appendCFunctionDefinition


class MetalProgram(
    private val context: MetalContext,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val library: MTLLibrary
    private val function: MTLFunction
    private val pipeline: MTLComputePipelineState
    private val commandEncoder: MTLComputeCommandEncoder

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)

        library = mtlCreateLibrary(context.device.peer, buffer.toString())
        function = mtlGetFunction(library, "_m")
        pipeline = mtlCreatePipeline(context.device.peer, function)
        commandEncoder = mtlCreateCommandEncoder(context.commandBuffer, pipeline)
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        buffers.forEachIndexed { i, field ->
            when(val value = map[field.name]!!){
                is Float -> mtlSetFloatAt(commandEncoder, value, i)
                is Int -> mtlSetIntAt(commandEncoder, value, i)
                is Byte -> mtlSetByteAt(commandEncoder, value, i)
                is MetalMemoryPointer<*> -> mtlSetBufferAt(commandEncoder, value.buffer, i)
                else -> throw UnsupportedOperationException()
            }
        }
        mtlSetIntAt(commandEncoder, indexOffset, buffers.size)
        mtlExecute(context.commandBuffer, commandEncoder, instances)
    }

    override fun dealloc() {
        mtlDeallocLibrary(library)
        mtlDeallocFunction(function)
        mtlDeallocPipeline(pipeline)
        mtlDeallocCommandEncoder(commandEncoder)
    }

    override fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: GPFunction) {
        buffer.append("kernel ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = "void",
            name = "_m",
            args = buffers.map {
                if(it.type.isArray) "device ${toCType(it.type)}*__v${it.name}"
                else "device ${toCType(it.type)}&__v${it.name}"
            } + listOf("device int&__o", "uint i [[thread_position_in_grid]]")
        )
    }
    override fun stringifyMainFunctionBody(buffer: StringBuilder, function: GPFunction) = Unit

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("M_PI")
            "E" -> buffer.append("M_E")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }

    override fun stringifyModifiersInStruct(field: GPField) =
        stringifyModifiersInArg(field)

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is GPField && obj.isConstant) "constant" else ""

    override fun stringifyModifiersInLocal(field: GPField) = ""

    override fun stringifyModifiersInArg(field: GPField) =
        if(field.type.isArray) "device" else ""
}
package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.CProgramPrinter


class MetalProgram(
    override val context: MetalContext,
    ast: GPScope
): GPProgram(ast) {
    override var released = false

    val library: MTLLibrary
    val function: MTLFunction
    val pipeline: MTLComputePipelineState
    val commandEncoder: MTLComputeCommandEncoder

    init {
        val prog = MetalProgramPrinter(ast, buffers, locals).stringify()

        library = mtlCreateLibrary(context.device.peer, prog)
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

    override fun release() {
        if(released) return
        context.releaseProgram(this)
        released = true
    }
}

class MetalProgramPrinter(
    ast: GPScope,
    buffers: List<GPField>,
    locals: List<GPField>
): CProgramPrinter(ast, buffers, locals){
    override fun stringifyMainFunctionDefinition(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) {
        buffer.append("kernel ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = "void",
            name = "_m",
            args = buffers.map {
                if (it.type.isArray) "device ${toCType(header, it.type)}*__v${it.obfName}"
                else "device ${toCType(header, it.type)}&__v${it.obfName}"
            } + listOf("device int&__o", "uint ${function.arguments[0].obfName} [[thread_position_in_grid]]")
        )
    }
    override fun stringifyMainFunctionBody(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) = Unit

    override fun stringifyFieldExpression(header: MutableMap<String, String>, buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("M_PI")
            "E" -> buffer.append("M_E")
            else -> super.stringifyFieldExpression(header, buffer, expression)
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
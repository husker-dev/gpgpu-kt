package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.CProgramPrinter
import kotlin.math.min


class MetalProgram(
    override val context: MetalContext,
    ast: GPScope
): GPProgram(ast) {
    override var released = false

    val library: MTLLibrary
    val function: MTLFunction
    val pipeline: MTLComputePipelineState
    val commandBuffer: MTLCommandBuffer
    val commandEncoder: MTLComputeCommandEncoder

    init {
        val prog = MetalProgramPrinter(ast, buffers, locals).stringify()

        library = mtlCreateLibrary(context.device.peer, prog)
        function = mtlGetFunction(library, "_m")
        pipeline = mtlCreatePipeline(context.device.peer, function)
        commandBuffer = mtlNewCommandBuffer(context.commandQueue)
        commandEncoder = mtlCreateCommandEncoder(commandBuffer, pipeline)
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        buffers.forEachIndexed { i, field ->
            when(val value = map[field.name]!!){
                is Float -> mtlSetFloatAt(commandEncoder, value, i)
                is Int -> mtlSetIntAt(commandEncoder, value, i)
                is Byte -> mtlSetByteAt(commandEncoder, value, i)
                is Boolean -> mtlSetByteAt(commandEncoder, if(value) 1 else 0, i)
                is MetalMemoryPointer<*> -> mtlSetBufferAt(commandEncoder, value.buffer, i)
                else -> throw UnsupportedOperationException()
            }
        }
        mtlSetIntAt(commandEncoder, indexOffset, buffers.size)

        val maxBlockDimX = maxTotalThreadsPerThreadgroup(pipeline)

        val blockSizeX = min(maxBlockDimX, instances)
        val gridSizeX = (instances + blockSizeX - 1) / blockSizeX

        mtlExecute(commandBuffer, commandEncoder, instances, gridSizeX)
    }

    override fun release() {
        if(released) return
        context.releaseProgram(this)
        mtlDeallocCommandEncoder(commandEncoder)
        mtlDeallocCommandBuffer(commandBuffer)
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
                if (it.type.isDynamicArray) "device ${toCType(header, it.type)}*__v${it.obfName}"
                else "device ${toCType(header, it.type)}&__v${it.obfName}"
            } + listOf("device int&__o", "uint ${function.arguments[0].obfName} [[thread_position_in_grid]]")
        )
    }
    override fun stringifyMainFunctionBody(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) = Unit

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

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) = when(val name = functionExpression.function.name) {
        "isNaN" -> "isnan"
        else -> "metal::$name"
    }

    override fun stringifyModifiersInStruct(field: GPField) =
        stringifyModifiersInArg(field)

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is GPField && obj.isConstant) "constant"
        else if(obj is GPFunction && obj.returnType.isDynamicArray) "device" else ""

    override fun stringifyModifiersInLocal(field: GPField) =
        if(field.type.isDynamicArray) "device" else ""

    override fun stringifyModifiersInArg(field: GPField) =
        stringifyModifiersInLocal(field)

    override fun stringifyModifiersInLocalsStruct() =
        "thread"
}
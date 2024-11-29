package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.ArrayPrimitiveType
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
    var commandBuffer: MTLCommandBuffer
    var commandEncoder: MTLComputeCommandEncoder

    val argumentEncoder: MTLArgumentEncoder
    val argumentBuffer: MTLBuffer

    init {
        val prog = MetalProgramPrinter(ast, buffers, locals).stringify()

        library = mtlCreateLibrary(context.device.peer, prog)
        function = mtlGetFunction(library, "_m")
        pipeline = mtlCreatePipeline(context.device.peer, function)
        commandBuffer = mtlNewCommandBuffer(context.commandQueue)
        commandEncoder = mtlCreateCommandEncoder(commandBuffer, pipeline)

        argumentEncoder = mtlCreateArgumentEncoderWithIndex(function, 0)
        argumentBuffer = mtlCreateAndBindArgumentBuffer(context.device.peer, argumentEncoder)
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        commandBuffer = mtlNewCommandBuffer(context.commandQueue)
        commandEncoder = mtlCreateCommandEncoder(commandBuffer, pipeline)
        mtlSetBufferAt(commandEncoder, argumentBuffer, 0)

        buffers.forEachIndexed { i, field ->
            when(val value = map[field.name]!!){
                is Float -> mtlSetFloatAt(argumentEncoder, value, i)
                is Int -> mtlSetIntAt(argumentEncoder, value, i)
                is Byte -> mtlSetByteAt(argumentEncoder, value, i)
                is Boolean -> mtlSetByteAt(argumentEncoder, if(value) 1 else 0, i)
                is MetalMemoryPointer<*> -> mtlSetBufferAt(argumentEncoder, value.buffer, i)
                else -> throw UnsupportedOperationException()
            }
        }
        mtlSetIntAt(argumentEncoder, indexOffset, buffers.size)

        val maxBlockDimX = maxTotalThreadsPerThreadgroup(pipeline)

        val blockSizeX = min(maxBlockDimX, instances)
        val gridSizeX = (instances + blockSizeX - 1) / blockSizeX

        mtlExecute(commandBuffer, commandEncoder, instances, gridSizeX)
        mtlRelease(commandEncoder)
        mtlRelease(commandBuffer)
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
): CProgramPrinter(ast, buffers, locals,
    useLocalStructCreation = false
){
    override fun stringifyScope(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        scope: GPScope,
        brackets: Boolean
    ) {
        if(scope.parentScope == null) {
            buffer.append("struct Arguments {")
            fun stringify(index: Int, field: GPField) {
                val modifiers = stringifyModifiersInStruct(field)
                if (modifiers.isNotEmpty())
                    buffer.append(modifiers).append(" ")

                buffer.append(toCType(header, field.type))
                    .append(" ")
                    .append(
                        if (field.type is ArrayPrimitiveType<*>)
                            convertArrayName(field.obfName, field.type.size)
                        else field.obfName
                    )
                buffer.append(" [[id(").append(index).append(")]];")
            }
            buffers.forEachIndexed { i, field -> stringify(i, field) }
            buffer.append("int offset;};")
        }
        super.stringifyScope(header, buffer, scope, brackets)
    }

    override fun stringifyMainFunctionDefinition(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) {
        buffer.append("kernel ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = "void",
            name = "_m",
            args = listOf("constant Arguments &args [[buffer(0)]]", "uint _i [[thread_position_in_grid]]")
        )
    }
    override fun stringifyMainFunctionBody(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) {
        buffer.append("int ").append(function.arguments[0].obfName).append("=_i+args.offset;")
        buffer.append("__in _v={")
        buffers.forEachIndexed { index, field ->
            buffer.append("args.").append(field.obfName)
            if (index != buffers.lastIndex || locals.isNotEmpty())
                buffer.append(",")
        }
        locals.forEachIndexed { index, field ->
            if(field.initialExpression != null)
                stringifyExpression(header, buffer, field.initialExpression!!, false)
            else buffer.append("0")
            if (index != buffers.lastIndex)
                buffer.append(",")
        }
        buffer.append("};")
        buffer.append("${stringifyModifiersInLocalsStruct()} __in*__v=&_v;")
    }

    override fun convertArrayName(name: String, size: Int) =
        if(size == -1) "* __restrict $name"
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
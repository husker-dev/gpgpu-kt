package com.huskerdev.gpkt.apis.webgpu

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.utils.SimpleCProgram

class WebGPUProgram(
    private val context: WebGPUAsyncContext,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val webgpu = context.webgpu

    private val groupLayout: dynamic
    private val shaderModule: dynamic
    private val pipeline: dynamic

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)
        println(buffer.toString())

        groupLayout = webgpu.createGroupLayout(context.devicePeer, buffers)
        shaderModule = webgpu.createShaderModule(context.devicePeer, buffer.toString())
        pipeline = webgpu.createPipeline(context.devicePeer, shaderModule, groupLayout, "_m")
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val arrays = buffers.map { field ->
            when(val value = map[field.name]!!){
                is WebGPUMemoryPointer<*> -> value
                else -> throw UnsupportedOperationException()
            }
        }
        webgpu.execute(
            context.devicePeer, context.commandEncoder,
            groupLayout, pipeline, arrays)
        context.flush()
    }

    override fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: GPFunction) {
        buffer.append("@compute@workgroup_size(1)fn _m(@builtin(global_invocation_id)id:vec3<u32>)")
    }

    override fun stringifyMainFunctionBody(buffer: StringBuilder, function: GPFunction) {
        buffer.append("let i=id.x;")
    }

    override fun stringifyModifiersInStruct(field: Field) = ""
    override fun stringifyModifiersInGlobal(obj: Any) = ""
    override fun stringifyModifiersInLocal(field: Field) = ""
    override fun stringifyModifiersInArg(field: Field) = ""

    override fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder) {
        val modifiers = fieldStatement.fields[0].modifiers
        val type = fieldStatement.fields[0].type

        if(Modifiers.EXTERNAL in modifiers){
            fieldStatement.fields.forEach { field ->
                val usage = if(field.isReadonly) "read" else "read_write"
                buffer.append("@group(0)@binding(")
                    .append(buffers.indexOf(field))
                    .append(")var<storage,").append(usage).append(">")
                    .append(field.name).append(":").append(toCType(type))
                    .append(";")
            }
        }else {
            fieldStatement.fields.forEachIndexed { i, field ->
                if(field.isConstant)
                    buffer.append("const")
                else buffer.append("var")
                buffer.append(" ").append(field.name).append(":").append(toCType(type)).append(";")
            }
        }
    }

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder) {
        val function = statement.function
        val name = function.name

        if(name != "main"){
            buffer.append("fn ").append(name).append("(")
            function.arguments.forEachIndexed { i, field ->
                buffer.append(field.name)
                    .append(":")
                    .append(toCType(field.type))
                if(i < function.arguments.lastIndex)
                    buffer.append(",")
            }
            buffer.append(")")
            if(function.returnType != VOID)
                buffer.append("->").append(toCType(function.returnType))
            stringifyScopeStatement(buffer, function.body, true)
        }else super.stringifyFunctionStatement(statement, buffer)
    }

    override fun stringifyConstExpression(buffer: StringBuilder, expression: ConstExpression) {
        buffer.append(expression.lexeme.text)
        if(expression.type == FLOAT && "." !in expression.lexeme.text)
            buffer.append(".0")
    }

    override fun stringifyAxBExpression(buffer: StringBuilder, expression: AxBExpression) {
        val leftType = expression.left.type
        val rightType = expression.right.type

        if(leftType != rightType){
            stringifyExpression(buffer, expression.left)
            buffer.append(expression.operator.token)
            buffer.append(toCType(leftType)).append("(")
            stringifyExpression(buffer, expression.right)
            buffer.append(")")
        }else super.stringifyAxBExpression(buffer, expression)
    }

    override fun toCType(type: PrimitiveType) = when(type){
        VOID -> throw UnsupportedOperationException()
        FLOAT -> "f32"
        INT -> "i32"
        BYTE -> TODO()
        BOOLEAN -> "bool"
        is FloatArrayType -> "array<f32>"
        is IntArrayType -> "array<i32>"
        is ByteArrayType -> TODO()
        else -> throw UnsupportedOperationException()
    }

    override fun dealloc() {
        TODO("Not yet implemented")
    }
}
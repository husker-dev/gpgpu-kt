package com.huskerdev.gpkt.engines.webgpu

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type

class WebGPUProgram(
    private val webgpu: WebGPU,
    ast: ScopeStatement
): SimpleCProgram(ast) {

    private val groupLayout: dynamic
    private val shaderModule: dynamic
    private val pipeline: dynamic

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)
        println(buffer.toString())

        groupLayout = webgpu.createGroupLayout(buffers)
        shaderModule = webgpu.createShaderModule(buffer.toString())
        pipeline = webgpu.createPipeline(shaderModule, groupLayout, "_m")
    }

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        val arrays = buffers.map { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            when(value){
                is WebGPUMemoryPointer<*> -> value
                else -> throw UnsupportedOperationException()
            }
        }
        webgpu.execute(groupLayout, pipeline, arrays)
    }

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

        if(name == "main"){
            buffer.append("@compute@workgroup_size(1)fn _m(@builtin(global_invocation_id)id:vec3<u32>){")
            buffer.append("let i=id.x;")
            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        }else {
            buffer.append("fn ").append(name).append("(")
            function.arguments.forEachIndexed { i, field ->
                buffer.append(field.name)
                    .append(":")
                    .append(toCType(field.type))
                if(i < function.arguments.lastIndex)
                    buffer.append(",")
            }
            buffer.append(")")
            if(function.returnType != Type.VOID)
                buffer.append("->").append(toCType(function.returnType))
            stringifyScopeStatement(buffer, function.body, true)
        }
    }

    override fun stringifyConstExpression(buffer: StringBuilder, expression: ConstExpression) {
        buffer.append(expression.lexeme.text)
        if(expression.type == Type.FLOAT && "." !in expression.lexeme.text)
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

    override fun toCType(type: Type) = when(type){
        Type.VOID -> throw UnsupportedOperationException()
        Type.FLOAT -> "f32"
        Type.DOUBLE -> "f32"
        Type.INT -> "i32"
        Type.BYTE -> TODO()
        Type.BOOLEAN -> "bool"
        Type.FLOAT_ARRAY -> "array<f32>"
        Type.DOUBLE_ARRAY -> "array<f32>"
        Type.INT_ARRAY -> "array<i32>"
        Type.BYTE_ARRAY -> TODO()
        Type.BOOLEAN_ARRAY -> "array<bool>"
    }

    override fun dealloc() {
        TODO("Not yet implemented")
    }
}
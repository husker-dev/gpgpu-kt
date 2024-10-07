package com.huskerdev.gpkt.engines.opengl

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.appendCFunctionHeader

class GLProgram(
    val openGL: OpenGL,
    val ast: ScopeStatement
): SimpleCProgram(ast) {

    private val program: Int

    init {
        val buffer = StringBuilder()
        buffer.append("#version 430 core\n")
        //buffer.append("#extension GL_NV_shader_buffer_load:enable\n")
        //buffer.append("#extension GL_NV_gpu_shader5:enable\n")
        buffer.append("layout(local_size_x=1,local_size_y=1,local_size_z=1)in;")

        // Declare buffers at first
        ast.statements.filter {
            it is FieldStatement && Modifiers.EXTERNAL in it.fields[0].modifiers
        }.forEach { st ->
            stringifyStatement(buffer, st)
        }

        // Declare remaining
        ast.statements.filter {
            it !is FieldStatement || Modifiers.EXTERNAL !in it.fields[0].modifiers
        }.forEach { st ->
            stringifyStatement(buffer, st)
        }


        println(buffer.toString().split(";", "\n").mapIndexed { index, s -> "${index+1}|\t$s" }.joinToString(";\n"))
        program = openGL.createProgram(buffer.toString().replace(";", ";\n"))
        /*
        program = openGL.createProgram("""
            #version 430 core
            #extension GL_NV_shader_buffer_load : enable
            layout(local_size_x=1,local_size_y=1,local_size_z=1) in;
            
            layout(std430,binding=0) buffer _data_{
                float data[];
            };
            layout(std430,binding=1) buffer _result_{
                float result[];
            };
            layout(location=2) uniform int minPeriod;
            layout(location=3) uniform int maxPeriod;
            layout(location=4) uniform int count;
            
            
            float sma(float *d, int from, int period){ 
                float sum=0;
                for(int i=0; i<period; i++)
                    if(from-i>=0)
                        sum+=d[from-i];
                return sum/float(period);
            }
            
            
            void main(){
                const int i = int(gl_GlobalInvocationID.x);
                int localPeriod = i/(maxPeriod-minPeriod)+minPeriod;
                int localCandle = i%(maxPeriod-minPeriod);
                
                result[i] = sma(data, localCandle,localPeriod);
                
            }
        """.trimIndent())

         */
    }

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        openGL.useProgram(program)

        buffers.forEachIndexed { i, field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            if(value !is GLMemoryPointer<*>) {
                when(value){
                    is Float -> openGL.setUniform1f(i, value)
                    is Int -> openGL.setUniform1i(i, value)
                    is Byte -> openGL.setUniform1b(i, value)
                    else -> throw UnsupportedOperationException()
                }
            }else
                openGL.setBufferIndex(i, value.ssbo)
        }
        // Set index offset variable
        //cl.setArgument(kernel, buffers.size, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(indexOffset)))
        openGL.launchProgram(instances)
    }

    override fun dealloc() =
        openGL.deallocProgram(program)

    override fun stringifyCastExpression(buffer: StringBuilder, expression: CastExpression) {
        buffer.append(toCType(expression.type)).append("(")
        stringifyExpression(buffer, expression.right)
        buffer.append(")")
    }

    override fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder) {
        val modifiers = fieldStatement.fields[0].modifiers
        if(Modifiers.EXTERNAL in modifiers){
            fieldStatement.fields.forEach { field ->
                if(field.type.isArray) {
                    buffer.append("layout(std430,binding=")
                        .append(buffers.indexOf(field))
                        .append(")buffer _")
                        .append(field.name)
                        .append("_{")
                        .append(toCType(field.type))
                        .append(" ")
                        .append(field.name)
                        .append("[];};")
                }else {
                    buffer.append("layout(location=")
                        .append(buffers.indexOf(field))
                        .append(")uniform ")
                        .append(toCType(field.type))
                        .append(" ")
                        .append(field.name)
                        .append(";")
                }
            }
        }else super.stringifyFieldStatement(fieldStatement, buffer)
    }

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder) {
        val function = statement.function
        if(function.name == "main"){
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = function.modifiers.map { it.text },
                type = toCType(function.returnType),
                name = function.name,
                args = emptyList()
            )
            buffer.append("{")
                .append("const int i=int(gl_GlobalInvocationID.x);")
            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        } else if(function.argumentsTypes.any { it.isArray }) {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = function.modifiers.map { it.text },
                type = toCType(function.returnType),
                name = function.name,
                args = function.arguments.map(::convertToFuncArg)
            )
            buffer.append("{")
            function.arguments
                .filter { it.type.isArray }
                .forEach { arg ->
                    buffer.append(toCType(arg.type))
                        .append(" ")
                        .append(toCArrayName(arg.name))
                        .append(";")

                    buffers.forEachIndexed { index, field ->
                        if(field.type.isArray){
                            buffer.append("if(__a_")
                                .append(arg.name)
                                .append("==")
                                .append(index)
                                .append(")")
                                .append(arg.name)
                                .append("=")
                                .append(field.name)
                                .append(";")
                        }
                    }
                }
            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        } else super.stringifyFunctionStatement(statement, buffer)
    }

    override fun stringifyFunctionCallExpression(buffer: StringBuilder, expression: FunctionCallExpression) {
        buffer.append(expression.function.name)
        buffer.append("(")
        expression.arguments.forEachIndexed { i, arg ->
            if(arg is FieldExpression && arg.type.isArray)
                buffer.append(buffers.indexOf(arg.field))
            else stringifyExpression(buffer, arg)

            if(i != expression.arguments.lastIndex)
                buffer.append(",")
        }
        buffer.append(")")
    }

    override fun convertToFuncArg(field: Field): String {
        return if(field.type.isArray){
            "int __a_${field.name}"
        } else super.convertToFuncArg(field)
    }

    override fun toCArrayName(name: String) =
        "$name[]"
}
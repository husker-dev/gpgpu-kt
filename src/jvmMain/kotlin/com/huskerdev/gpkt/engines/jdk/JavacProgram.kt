package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import com.huskerdev.gpkt.utils.splitThreadInvocation
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.util.concurrent.atomic.AtomicLong


class JavacProgram(ast: Scope): SimpleCProgram(ast) {

    companion object {
        val counter = AtomicLong()
    }

    private val execMethod: Method

    init {
        val className = "GPJavacProgram${counter.getAndIncrement()}"
        val buffer = StringBuilder()
        buffer.append("""
            public class $className{ 
                public static void _execute(int fromIndex, int toIndex, ${buffers.joinToString { "float[] $it" }}){
                    for(int i = fromIndex; i < toIndex; i++)
                        __m(${(arrayOf("i") + buffers).joinToString(",")});
                }
                
        """.trimIndent())
        stringifyScope(ast, buffer)
        buffer.append("}")

        val clazz = ClassCompiler.compileClass(buffer.toString(), className)
        execMethod = clazz.getMethod(
            "_execute",
            *(arrayListOf(Int::class.java, Int::class.java) + buffers.map { FloatArray::class.java }).toTypedArray()
        )
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val map = hashMapOf(*mapping)

        val arrays = buffers.map {
            if(it !in map) throw Exception("Source '$it' have not been set")
            else (map[it] as JavacSource).array
        }

        splitThreadInvocation(instances) { from, to ->
            try {
                execMethod.invoke(null, *(listOf(from, to) + arrays).toTypedArray())
            }catch (e: InvocationTargetException){
                throw e.targetException
            }
        }
    }

    override fun dealloc() = Unit

    override fun stringifyFunction(function: Function, buffer: StringBuilder){
        val funName = if(function.name == "main") "__m" else function.name
        val args = if(function.name == "main")
            listOf("int i") + buffers.map { "float[] $it" }
        else
            function.arguments.map { "${it.type.toCType(false)} ${it.name}" }

        appendCFunctionHeader(
            buffer = buffer,
            modifiers = listOf("private", "static") + function.modifiers.map { it.text },
            type = function.returnType.text,
            name = funName,
            args = args
        )
        stringifyScope(function, buffer)
        buffer.append("}")
    }

    override fun Type.toCType(isGlobal: Boolean): String {
        val modifiers = if(isGlobal) "private static " else ""
        return modifiers + when (this) {
            Type.VOID -> "void"
            Type.FLOAT -> "float"
            Type.INT -> "int"
            Type.BOOLEAN -> "bool"
            Type.FLOAT_ARRAY -> "float[]"
            Type.INT_ARRAY -> "int[]"
            Type.BOOLEAN_ARRAY -> "bool[]"
        }
    }
}
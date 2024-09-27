package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.engines.cpu.*
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
                public static void _execute(int fromIndex, int toIndex, ${buffers.joinToString { toCType(it.type, false, it.name) }}){
                    for(int i = fromIndex; i < toIndex; i++)
                        __m(${(arrayOf("i") + buffers.map { it.name }).joinToString(",")});
                }
        """.trimIndent())
        stringifyScope(ast, buffer)
        buffer.append("}")

        val clazz = ClassCompiler.compileClass(buffer.toString(), className)
        execMethod = clazz.getMethod(
            "_execute",
            Int::class.java, Int::class.java,
            *(buffers.map { it.type.toJavaClass() }.toTypedArray())
        )
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        val arrays = buffers.map { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            when(value){
                is CPUFloatMemoryPointer -> value.array
                is CPUDoubleMemoryPointer -> value.array
                is CPULongMemoryPointer -> value.array
                is CPUIntMemoryPointer -> value.array
                is CPUByteMemoryPointer -> value.array
                is Float, Double, Long, Int, Byte -> value
                else -> UnsupportedOperationException()
            }
        }

        splitThreadInvocation(instances) { from, to ->
            try {
                execMethod.invoke(null, from, to, *arrays.toTypedArray())
            }catch (e: InvocationTargetException){
                throw e.targetException
            }
        }
    }

    override fun dealloc() = Unit

    override fun stringifyFunction(function: Function, buffer: StringBuilder){
        val funName = if(function.name == "main") "__m" else function.name
        val args = if(function.name == "main")
            listOf("int i") + buffers.map { toCType(it.type, false, it.name) }
        else
            function.arguments.map { toCType(it.type, false, it.name) }

        appendCFunctionHeader(
            buffer = buffer,
            modifiers = listOf("private", "static", "final") + function.modifiers.map { it.text },
            type = function.returnType.text,
            name = funName,
            args = args
        )
        stringifyScope(function, buffer)
        buffer.append("}")
    }

    private fun Type.toJavaClass() = when(this) {
        Type.VOID -> Unit::class.java
        Type.FLOAT -> Float::class.java
        Type.DOUBLE -> Double::class.java
        Type.LONG -> Long::class.java
        Type.INT -> Int::class.java
        Type.BYTE -> Byte::class.java
        Type.BOOLEAN -> Boolean::class.java
        Type.FLOAT_ARRAY -> FloatArray::class.java
        Type.DOUBLE_ARRAY -> DoubleArray::class.java
        Type.LONG_ARRAY -> LongArray::class.java
        Type.INT_ARRAY -> IntArray::class.java
        Type.BYTE_ARRAY -> ByteArray::class.java
        Type.BOOLEAN_ARRAY -> BooleanArray::class.java
    }

    override fun toCType(type: Type, isConst: Boolean, name: String): String {
        val modifiers = if(isConst) "private static final " else ""
        return modifiers + when (type) {
            Type.VOID -> if(name.isEmpty()) "void" else "void $name"
            Type.FLOAT -> if(name.isEmpty()) "float" else "float $name"
            Type.LONG -> if(name.isEmpty()) "long" else "long$name"
            Type.INT -> if(name.isEmpty()) "int" else "int $name"
            Type.DOUBLE -> if(name.isEmpty()) "double" else "double $name"
            Type.BYTE -> if(name.isEmpty()) "byte" else "byte $name"
            Type.BOOLEAN -> if(name.isEmpty()) "bool" else "bool $name"
            Type.FLOAT_ARRAY -> if(name.isEmpty()) "float[]" else "float[]$name"
            Type.LONG_ARRAY -> if(name.isEmpty()) "long[]" else "long[]$name"
            Type.INT_ARRAY -> if(name.isEmpty()) "int[]" else "int[]$name"
            Type.DOUBLE_ARRAY -> if(name.isEmpty()) "double[]" else "double[]$name"
            Type.BYTE_ARRAY -> if(name.isEmpty()) "byte[]" else "byte[]$name"
            Type.BOOLEAN_ARRAY -> if(name.isEmpty()) "boolean[]" else "boolean[]$name"
        }
    }
}
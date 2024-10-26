package com.huskerdev.gpkt.apis.jdk

import com.huskerdev.gpkt.apis.interpreter.CPUMemoryPointer
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.utils.SimpleCProgram
import com.huskerdev.gpkt.utils.appendCFunctionDefinition
import com.huskerdev.gpkt.utils.splitThreadInvocation
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.util.concurrent.atomic.AtomicLong


class JavacProgram(ast: ScopeStatement): SimpleCProgram(ast, false, false, false) {
    companion object {
        val counter = AtomicLong()
    }

    private val execMethod: Method

    init {
        val className = "GPJavacProgram${counter.getAndIncrement()}"
        val buffer = StringBuilder()
        buffer.append("""
            import static java.lang.Math.*;
            public class $className{ 
                public static void _execute(int fromIndex, int toIndex, ${buffers.joinToString{ "${toCType(it.type)} __v${it.name}" }}){
                    ${buffers.joinToString("") { "${it.name}=__v${it.name};" }}
                    for(int i = fromIndex; i < toIndex; i++)
                        _m(i);
                }
                private static int _aRead(int[] arr, int i){
                    if(i < 0 || i > arr.length-1) return 0;
                    else return arr[i];
                }
                private static float _aRead(float[] arr, int i){
                    if(i < 0 || i > arr.length-1) return 0f;
                    else return arr[i];
                }
                private static byte _aRead(byte[] arr, int i){
                    if(i < 0 || i > arr.length-1) return 0;
                    else return arr[i];
                }
                private static boolean _aRead(boolean[] arr, int i){
                    if(i < 0 || i > arr.length-1) return false;
                    else return arr[i];
                }
                private static void _aSet(int[] arr, int i, int value){
                    if(i < 0 || i > arr.length-1) return;
                    else arr[i] = value;
                }
                private static void _aSet(float[] arr, int i, float value){
                    if(i < 0 || i > arr.length-1) return;
                    else arr[i] = value;
                }
                private static void _aSet(byte[] arr, int i, byte value){
                    if(i < 0 || i > arr.length-1) return;
                    else arr[i] = value;
                }
                private static void _aSet(boolean[] arr, int i, boolean value){
                    if(i < 0 || i > arr.length-1) return;
                    else arr[i] = value;
                }
        """.trimIndent())
        stringifyScopeStatement(buffer, ast, false)
        buffer.append("}")

        val clazz = ClassCompiler.compileClass(buffer.toString(), className)
        execMethod = clazz.getMethod(
            "_execute",
            Int::class.java, Int::class.java,
            *(buffers.map { it.type.toJavaClass() }.toTypedArray())
        )
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val arrays = buffers.map { field ->
            when(val value = map[field.name]!!){
                is CPUMemoryPointer<*> -> value.array
                is Float, is Double, is Long, is Int, is Byte -> value
                else -> throw UnsupportedOperationException()
            }
        }

        splitThreadInvocation(instances) { from, to ->
            try {
                execMethod.invoke(null,
                    from + indexOffset,
                    to + indexOffset,
                    *arrays.toTypedArray()
                )
            }catch (e: InvocationTargetException){
                throw e.targetException
            }
        }
    }

    override fun dealloc() = Unit

    override fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: GPFunction) {
        buffer.append("private static final ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.toString(),
            name = "_m",
            args = listOf("int ${function.arguments[0].name}")
        )
    }

    override fun stringifyMainFunctionBody(buffer: StringBuilder, function: GPFunction) = Unit

    override fun stringifyModifiersInStruct(field: Field) = ""

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is Field && obj.isConstant) "private static final"
        else "private static"

    override fun stringifyModifiersInLocal(field: Field) =
        if(field.isConstant) "final"
        else ""

    override fun stringifyModifiersInArg(field: Field) = ""

    override fun stringifyAxBExpression(buffer: StringBuilder, expression: AxBExpression) {
        if(expression.operator == Operator.ASSIGN && expression.left is ArrayAccessExpression){
            buffer.append("_aSet(")
            stringifyExpression(buffer, expression.left.array)
            buffer.append(",")
            stringifyExpression(buffer, expression.left.index)
            buffer.append(",")
            stringifyExpression(buffer, expression.right)
            buffer.append(")")
        }else super.stringifyAxBExpression(buffer, expression)
    }

    override fun stringifyArrayAccessExpression(buffer: StringBuilder, expression: ArrayAccessExpression) {
        buffer.append("_aRead(")
        stringifyExpression(buffer, expression.array)
        buffer.append(",")
        stringifyExpression(buffer, expression.index)
        buffer.append(")")
    }

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(val name = expression.field.name){
            "E", "PI" -> buffer.append("(float)$name")
            else -> buffer.append(name)
        }
    }

    override fun stringifyArrayDefinitionExpression(buffer: StringBuilder, expression: ArrayDefinitionExpression) {
        buffer.append("new ").append(toCType(expression.type))
        super.stringifyArrayDefinitionExpression(buffer, expression)
    }

    override fun toCType(type: PrimitiveType) = when (type) {
        VOID -> "void"
        FLOAT -> "float"
        INT -> "int"
        BYTE -> "byte"
        BOOLEAN -> "boolean"
        is FloatArrayType -> "float[]"
        is IntArrayType -> "int[]"
        is ByteArrayType -> "byte[]"
        else -> throw UnsupportedOperationException()
    }

    override fun toCArrayName(name: String, size: Int) = name

    private fun PrimitiveType.toJavaClass() = when(this) {
        VOID -> Unit::class.java
        FLOAT -> Float::class.java
        INT -> Int::class.java
        BYTE -> Byte::class.java
        BOOLEAN -> Boolean::class.java
        is FloatArrayType -> FloatArray::class.java
        is IntArrayType -> IntArray::class.java
        is ByteArrayType -> ByteArray::class.java
        else -> throw UnsupportedOperationException()
    }
}
package com.huskerdev.gpkt.apis.jdk

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.apis.interpreter.CPUMemoryPointer
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.predefinedMathFunctions
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import com.huskerdev.gpkt.utils.splitThreadInvocation
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.util.concurrent.atomic.AtomicLong


class JavacProgram(ast: ScopeStatement): SimpleCProgram(ast) {
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
                public static void _execute(int fromIndex, int toIndex, ${buffers.joinToString(transform = ::transformKernelArg)}){
                    ${buffers.joinToString("") { "${it.name}=__v_${it.name};" }}
                    for(int i = fromIndex; i < toIndex; i++)
                        __m(i);
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

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        val arrays = buffers.map { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            when(value){
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

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        val name: String
        val args: List<String>
        if(function.name == "main"){
            name = "__m"
            args = listOf("int i")
        }else {
            name = function.name
            args = function.arguments.map(::convertToFuncArg)
        }
        appendCFunctionHeader(
            buffer = buffer,
            modifiers = listOf("private", "static", "final") + function.modifiers.map { it.text },
            type = function.returnType.text,
            name = name,
            args = args
        )
        stringifyScopeStatement(buffer, function.body, true)
    }

    override fun stringifyAxBExpression(buffer: StringBuilder, expression: AxBExpression) {
        if(expression.operator == Operator.ASSIGN && expression.left is ArrayAccessExpression){
            buffer.append("_aSet(")
                .append(expression.left.array.name)
                .append(",")
            stringifyExpression(buffer, expression.left.index)
            buffer.append(",")
            stringifyExpression(buffer, expression.right)
            buffer.append(")")
        }else super.stringifyAxBExpression(buffer, expression)
    }

    override fun stringifyArrayAccessExpression(buffer: StringBuilder, expression: ArrayAccessExpression) {
        buffer.append("_aRead(")
        buffer.append(expression.array.name)
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

    override fun stringifyFunctionCallExpression(buffer: StringBuilder, expression: FunctionCallExpression) {
        if(expression.function.name in predefinedMathFunctions && expression.function.returnType.isFloating)
            buffer.append("(float)")
        super.stringifyFunctionCallExpression(buffer, expression)
    }

    private fun transformKernelArg(field: Field) =
        "${toCType(field.type)} __v_${field.name}"

    private fun Type.toJavaClass() = when(this) {
        Type.VOID -> Unit::class.java
        Type.FLOAT -> Float::class.java
        Type.INT -> Int::class.java
        Type.BYTE -> Byte::class.java
        Type.BOOLEAN -> Boolean::class.java
        Type.FLOAT_ARRAY -> FloatArray::class.java
        Type.INT_ARRAY -> IntArray::class.java
        Type.BYTE_ARRAY -> ByteArray::class.java
        Type.BOOLEAN_ARRAY -> BooleanArray::class.java
    }

    override fun toCType(type: Type) = when (type) {
        Type.VOID -> "void"
        Type.FLOAT -> "float"
        Type.INT -> "int"
        Type.BYTE -> "byte"
        Type.BOOLEAN -> "boolean"
        Type.FLOAT_ARRAY -> "float[]"
        Type.INT_ARRAY -> "int[]"
        Type.BYTE_ARRAY -> "byte[]"
        Type.BOOLEAN_ARRAY -> "boolean[]"
    }

    override fun toCArrayName(name: String) = name

    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.CONST -> "private static final"
        Modifiers.EXTERNAL -> "private static"
        Modifiers.READONLY -> ""
    }
}
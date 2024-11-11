package com.huskerdev.gpkt.apis.jdk

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.apis.interpreter.CPUMemoryPointer
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.utils.CProgramPrinter
import com.huskerdev.gpkt.utils.splitThreadInvocation
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.util.concurrent.atomic.AtomicLong


class JavacProgram(ast: GPScope): GPProgram(ast) {
    companion object {
        val counter = AtomicLong()
    }

    private val execMethod: Method

    init {
        val className = "GPJavacProgram${counter.getAndIncrement()}"
        val prog = JavacProgramPrinter(className, ast, buffers, locals).stringify()

        val clazz = ClassCompiler.compileClass(prog, className)
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
                is Float, is Double, is Long, is Int, is Byte, is Boolean -> value
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

private class JavacProgramPrinter(
    val className: String,
    ast: GPScope,
    buffers: List<GPField>,
    locals: List<GPField>
): CProgramPrinter(ast, buffers, locals,
    useLocalStruct = false,
    useArrayStruct = false,
    useArrayStructCast = false,
    useFunctionDefs = false,
    useStructClasses = false
){
    override fun stringify() = """
        import static java.lang.Math.*;
        public class $className{ 
            public static void _execute(int fromIndex, int toIndex, ${buffers.joinToString{ "${convertType(it.type)} __v${it.obfName}" }}){
                ${buffers.joinToString("") { "${it.obfName}=__v${it.obfName};" }}
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
            ${super.stringify()}
        }
    """.trimIndent()


    override fun stringifyMainFunctionDefinition(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("private static final ")
        com.huskerdev.gpkt.utils.appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.toString(),
            name = "_m",
            args = listOf("int ${function.arguments[0].obfName}")
        )
    }

    override fun stringifyMainFunctionBody(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) = Unit

    override fun stringifyModifiersInStruct(field: GPField) = ""

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is GPFunction && obj.scope!!.parentScope != null) "public"
        else if(obj is GPField && obj.isConstant) "private static final"
        else "private static"

    override fun stringifyModifiersInLocal(field: GPField) =
        if(field.isConstant) "final"
        else ""

    override fun stringifyModifiersInArg(field: GPField) = ""

    override fun stringifyAxBExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: AxBExpression
    ) {
        if(expression.operator == Operator.ASSIGN && expression.left is ArrayAccessExpression){
            buffer.append("_aSet(")
            stringifyExpression(header, buffer, expression.left.array)
            buffer.append(",")
            stringifyExpression(header, buffer, expression.left.index)
            buffer.append(",")
            stringifyExpression(header, buffer, expression.right)
            buffer.append(")")
        }else super.stringifyAxBExpression(header, buffer, expression)
    }

    override fun stringifyArrayAccessExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ArrayAccessExpression
    ) {
        buffer.append("_aRead(")
        stringifyExpression(header, buffer, expression.array)
        buffer.append(",")
        stringifyExpression(header, buffer, expression.index)
        buffer.append(")")
    }

    override fun convertPredefinedFieldName(field: GPField) = when(field.name){
        "E", "PI" -> "(float)${field.name}"
        "NaN" -> "Float.NaN"
        else -> field.obfName
    }

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) = when(functionExpression.function.name){
        "isNaN" -> "Float.isNaN"
        else -> functionExpression.function.obfName
    }

    override fun stringifyArrayDefinitionExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ArrayDefinitionExpression
    ) {
        buffer.append("new ").append(toCType(header, expression.type))
        super.stringifyArrayDefinitionExpression(header, buffer, expression)
    }

    override fun stringifyClassStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        classStatement: ClassStatement
    ) {
        val clazz = classStatement.classObj

        buffer.append("private static class ").append(clazz.obfName).append("{")
        clazz.variables.values.joinTo(buffer, separator = ";"){
            convertToFuncArg(header, it)
        }
        if(clazz.variables.isNotEmpty())
            buffer.append(";")

        // Constructor
        buffer.append("public ").append(clazz.obfName).append("(")
        clazz.variables.values.joinTo(buffer, separator = ","){
            convertToFuncArg(header, it)
        }
        buffer.append("){")
        clazz.variables.values.joinTo(buffer, separator = ";"){
            "this.${it.obfName}=${it.obfName}"
        }
        if(clazz.variables.isNotEmpty())
            buffer.append(";")
        buffer.append("}")

        clazz.body?.scopeObj?.statements?.forEach {
            if(it is FunctionStatement)
                stringifyFunctionStatement(header, buffer, it)
        }
        buffer.append("}")
    }

    override fun stringifyClassCreationExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ClassCreationExpression
    ) {
        buffer.append("new ").append(expression.classObj.obfName).append("(")
        expression.arguments.forEachIndexed { i, e ->
            val targetType = expression.classObj.variablesTypes[i]
            if(e.type != targetType)
                buffer.append("(").append(convertType(targetType)).append(")")
            stringifyExpression(header, buffer, e)
            if (i < expression.arguments.size - 1)
                buffer.append(",")
        }
        buffer.append(")")
    }

    override fun convertType(type: PrimitiveType) = when (type) {
        VOID -> "void"
        FLOAT -> "float"
        INT -> "int"
        BYTE -> "byte"
        BOOLEAN -> "boolean"
        is FloatArrayType -> "float[]"
        is IntArrayType -> "int[]"
        is ByteArrayType -> "byte[]"
        else -> type.toString()
    }

    override fun convertArrayName(name: String, size: Int) = name
}
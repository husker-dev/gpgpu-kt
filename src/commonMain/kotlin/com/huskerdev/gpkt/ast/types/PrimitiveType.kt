package com.huskerdev.gpkt.ast.types


val DYNAMIC_FLOAT_ARRAY = FloatArrayType(-1)
val DYNAMIC_INT_ARRAY = IntArrayType(-1)
val DYNAMIC_BYTE_ARRAY = ByteArrayType(-1)

val VOID = VoidType
val FLOAT = FloatType
val INT = IntType
val BYTE = ByteType
val BOOLEAN = BooleanType

val primitivesMap = hashMapOf<String, SinglePrimitiveType<*>>(
    "void" to VOID,
    "float" to FLOAT,
    "int" to INT,
    "byte" to BYTE,
    "boolean" to BOOLEAN,
)

interface PrimitiveType {
    val bytes: Int
    val isNumber: Boolean
        get() = isInteger || isFloating
    val isFloating: Boolean
    val isInteger: Boolean
    val isLogical: Boolean
    val isArray: Boolean
    val isDynamicArray: Boolean
    val isConstArray: Boolean

    companion object {
        val allowedCastMap = mapOf(
            FLOAT to setOf(FLOAT, INT, BYTE),
            INT to setOf(FLOAT, INT, BYTE),
            BYTE to setOf(FLOAT, INT, BYTE),
        )

        fun canAssignNumbers(to: PrimitiveType, from: PrimitiveType): Boolean {
            return if(to == BYTE && from == INT)
                true // add check fort value 127
            else {
                to.isNumber && from.isNumber &&
                (to.isFloating || to.isFloating == from.isFloating) &&
                to.bytes >= from.bytes
            }
        }

        fun mergeNumberTypes(type1: SinglePrimitiveType<*>, type2: SinglePrimitiveType<*>) = when {
            type1 is ClassType && type2 !is ClassType -> type2
            type1 !is ClassType && type2 is ClassType -> type1
            type1 == type2 -> type1
            type1.isFloating && !type2.isFloating -> type1
            !type1.isFloating && type2.isFloating -> type2
            type1.bytes > type2.bytes -> type1
            else -> type2
        }
    }
}

interface SinglePrimitiveType<A: PrimitiveType>: PrimitiveType {
    val toArray: (Int) -> A
    val toDynamicArray: () -> A
}

interface ArrayPrimitiveType<S: PrimitiveType>: PrimitiveType {
    val size: Int
    val single: S
}

interface ClassType: SinglePrimitiveType<Nothing> {
    val className: String
    val classNameObf: String
}

object VoidType: SinglePrimitiveType<Nothing> {
    override val bytes = 0
    override val isFloating = false
    override val isInteger = false
    override val isLogical = false
    override val isArray = false
    override val isDynamicArray = false
    override val isConstArray = false
    override val toArray = { _: Int -> throw UnsupportedOperationException() }
    override val toDynamicArray = { throw UnsupportedOperationException() }
    override fun toString() = "void"
}

object IntType: SinglePrimitiveType<IntArrayType>{
    override val bytes = 4
    override val isFloating = false
    override val isInteger = true
    override val isLogical = false
    override val isArray = false
    override val isDynamicArray = false
    override val isConstArray = false
    override val toArray = ::IntArrayType
    override val toDynamicArray = ::DYNAMIC_INT_ARRAY
    override fun toString() = "int"
}

object FloatType: SinglePrimitiveType<FloatArrayType>{
    override val bytes = 4
    override val isFloating = true
    override val isInteger = false
    override val isLogical = false
    override val isArray = false
    override val isDynamicArray = false
    override val isConstArray = false
    override val toArray = ::FloatArrayType
    override val toDynamicArray = ::DYNAMIC_FLOAT_ARRAY
    override fun toString() = "float"
}

object ByteType: SinglePrimitiveType<ByteArrayType>{
    override val bytes = 1
    override val isFloating = false
    override val isInteger = true
    override val isLogical = false
    override val isArray = false
    override val isDynamicArray = false
    override val isConstArray = false
    override val toArray = ::ByteArrayType
    override val toDynamicArray = ::DYNAMIC_BYTE_ARRAY
    override fun toString() = "byte"
}

object BooleanType: SinglePrimitiveType<Nothing>{
    override val bytes = 1
    override val isFloating = false
    override val isInteger = false
    override val isLogical = true
    override val isArray = false
    override val isDynamicArray = false
    override val isConstArray = false
    override val toArray = { _: Int -> throw UnsupportedOperationException() }
    override val toDynamicArray = { throw UnsupportedOperationException() }
    override fun toString() = "boolean"
}

data class IntArrayType(
    override val size: Int
): ArrayPrimitiveType<IntType>{
    override val single = IntType
    override val bytes = size * 4
    override val isFloating = false
    override val isInteger = false
    override val isLogical = false
    override val isArray = true
    override val isConstArray = size != -1
    override val isDynamicArray = size == -1
    override fun toString() = if(isDynamicArray) "int[]" else "int[$size]"
}

data class FloatArrayType(
    override val size: Int
): ArrayPrimitiveType<FloatType>{
    override val single = FloatType
    override val bytes = size * 4
    override val isFloating = false
    override val isInteger = false
    override val isLogical = false
    override val isArray = true
    override val isConstArray = size != -1
    override val isDynamicArray = size == -1
    override fun toString() = if(isDynamicArray) "float[]" else "float[$size]"
}

data class ByteArrayType(
    override val size: Int
): ArrayPrimitiveType<ByteType>{
    override val single = ByteType
    override val bytes = size * 1
    override val isFloating = false
    override val isInteger = false
    override val isLogical = false
    override val isArray = true
    override val isConstArray = size != -1
    override val isDynamicArray = size == -1
    override fun toString() = if(isDynamicArray) "byte[]" else "byte[$size]"
}
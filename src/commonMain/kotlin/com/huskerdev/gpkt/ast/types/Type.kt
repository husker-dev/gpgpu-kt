package com.huskerdev.gpkt.ast.types

const val FLAG_NUMBER = 1
const val FLAG_LOGICAL = 2
const val FLAG_ARRAY = 4

enum class Type(
    val text: String,
    flags: Int
) {
    VOID("void", 0),

    FLOAT("float", FLAG_NUMBER),
    INT("int", FLAG_NUMBER),
    BOOLEAN("boolean", FLAG_LOGICAL),

    FLOAT_ARRAY("float[]", FLAG_ARRAY),
    INT_ARRAY("int[]", FLAG_ARRAY),
    BOOLEAN_ARRAY("boolean[]", FLAG_ARRAY)
    ;

    val isNumber = flags and FLAG_NUMBER == FLAG_NUMBER
    val isLogical = flags and FLAG_LOGICAL == FLAG_LOGICAL
    val isArray = flags and FLAG_ARRAY == FLAG_ARRAY

    companion object {
        val map = entries.associateBy { it.text }

        val castMap = mapOf(
            FLOAT to setOf(FLOAT, INT),
            INT to setOf(FLOAT, INT)
        )

        fun toArrayType(type: Type) = when(type){
            FLOAT -> FLOAT_ARRAY
            INT -> INT_ARRAY
            BOOLEAN -> BOOLEAN_ARRAY
            else -> throw UnsupportedOperationException()
        }

        fun toSingleType(type: Type) = when(type){
            FLOAT_ARRAY -> FLOAT
            INT_ARRAY -> INT
            BOOLEAN_ARRAY -> BOOLEAN
            else -> throw UnsupportedOperationException()
        }
    }
}
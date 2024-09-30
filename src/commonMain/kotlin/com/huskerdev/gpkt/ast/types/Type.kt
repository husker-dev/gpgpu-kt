package com.huskerdev.gpkt.ast.types

const val FLAG_NUMBER = 1
const val FLAG_FLOATING_POINT = 2
const val FLAG_INTEGER = 3
const val FLAG_LOGICAL = 4
const val FLAG_ARRAY = 5

enum class Type(
    val text: String,
    val bytes: Int,
    flags: Int,
) {
    VOID("void", -1, 0),

    FLOAT("float", 4, FLAG_NUMBER or FLAG_FLOATING_POINT),
    DOUBLE("double", 8, FLAG_NUMBER or FLAG_FLOATING_POINT),
    LONG("long", 8, FLAG_NUMBER or FLAG_INTEGER),
    INT("int", 4, FLAG_NUMBER or FLAG_INTEGER),
    BYTE("byte", 1, FLAG_NUMBER or FLAG_INTEGER),
    BOOLEAN("boolean", 1, FLAG_LOGICAL),

    FLOAT_ARRAY("float[]", -1, FLAG_ARRAY),
    DOUBLE_ARRAY("double[]", -1, FLAG_ARRAY),
    LONG_ARRAY("long[]", -1, FLAG_ARRAY),
    INT_ARRAY("int[]", -1, FLAG_ARRAY),
    BYTE_ARRAY("byte[]", -1, FLAG_ARRAY),
    BOOLEAN_ARRAY("boolean[]", -1, FLAG_ARRAY)
    ;

    val isNumber = flags and FLAG_NUMBER == FLAG_NUMBER
    val isFloating = flags and FLAG_FLOATING_POINT == FLAG_FLOATING_POINT
    val isInteger = flags and FLAG_INTEGER == FLAG_INTEGER
    val isLogical = flags and FLAG_LOGICAL == FLAG_LOGICAL
    val isArray = flags and FLAG_ARRAY == FLAG_ARRAY

    companion object {
        val map = entries.associateBy { it.text }

        val allowedCastMap = mapOf(
            FLOAT to setOf(DOUBLE, FLOAT, INT, BYTE),
            DOUBLE to setOf(DOUBLE, FLOAT, INT, BYTE),
            LONG to setOf(DOUBLE, FLOAT, INT, BYTE),
            INT to setOf(DOUBLE, FLOAT, INT, BYTE),
            BYTE to setOf(DOUBLE, FLOAT, INT, BYTE),
        )

        fun toArrayType(type: Type) = when(type){
            FLOAT -> FLOAT_ARRAY
            DOUBLE -> DOUBLE_ARRAY
            LONG -> LONG_ARRAY
            INT -> INT_ARRAY
            BYTE -> BYTE_ARRAY
            BOOLEAN -> BOOLEAN_ARRAY
            else -> throw UnsupportedOperationException()
        }

        fun toSingleType(type: Type) = when(type){
            DOUBLE_ARRAY -> DOUBLE
            FLOAT_ARRAY -> FLOAT
            LONG_ARRAY -> LONG
            INT_ARRAY -> INT
            BYTE_ARRAY -> BYTE
            BOOLEAN_ARRAY -> BOOLEAN
            else -> throw UnsupportedOperationException()
        }

        fun canAssignNumbers(to: Type, from: Type) =
            to.isNumber && from.isNumber && to.bytes >= from.bytes

        fun mergeNumberTypes(type1: Type, type2: Type) = when {
            type1 == type2 -> type1
            type1.isFloating && !type2.isFloating -> type1
            !type1.isFloating && type2.isFloating -> type2
            type1.bytes > type1.bytes -> type1
            else -> type2
        }

    }
}
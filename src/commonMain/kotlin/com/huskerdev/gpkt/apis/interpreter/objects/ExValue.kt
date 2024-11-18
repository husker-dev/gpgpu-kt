package com.huskerdev.gpkt.apis.interpreter.objects

import com.huskerdev.gpkt.ast.types.BYTE
import com.huskerdev.gpkt.ast.types.FLOAT
import com.huskerdev.gpkt.ast.types.INT
import com.huskerdev.gpkt.ast.types.PrimitiveType


open class ExValue(
    private var value: Any?
) {
    open fun set(newValue: Any?): ExValue {
        value = newValue
        return this
    }
    open fun get() = value

    fun castToType(type: PrimitiveType?) = when{
        type == FLOAT && value !is Float -> ExValue((get() as Number).toFloat())
        type == INT && value !is Double -> ExValue((get() as Number).toInt())
        type == BYTE && value !is Byte -> ExValue((get() as Number).toByte())
        else -> this
    }

}

class ExArrayAccessValue(
    val array: Any,
    var index: Int
): ExValue(null) {
    private val length = when(array){
        is FloatArray -> array.size
        is IntArray -> array.size
        is ByteArray -> array.size
        is BooleanArray -> array.size
        else -> throw UnsupportedOperationException()
    }

    override fun set(newValue: Any?): ExValue {
        if(index < 0 || index >= length)
            return ExValue(0)
        when (array) {
            is FloatArray -> array[index] = newValue as Float
            is IntArray -> array[index] = newValue as Int
            is ByteArray -> array[index] = when (newValue) {
                is Byte -> newValue
                is Int -> newValue.toByte()
                else -> throw UnsupportedOperationException()
            }
            is BooleanArray -> array[index] = newValue as Boolean
            else -> throw UnsupportedOperationException("Can't set element in $array")
        }
        return this
    }

    override fun get(): Any = when (array) {
        is FloatArray -> array.getOrElse(index) { 0f }
        is IntArray -> array.getOrElse(index) { 0 }
        is ByteArray -> array.getOrElse(index) { 0.toByte() }
        is BooleanArray -> array.getOrElse(index) { false }
        else -> throw UnsupportedOperationException("Can't get element in $array")
    }
}

object BreakMarker: ExValue(null)
object ContinueMarker: ExValue(null)
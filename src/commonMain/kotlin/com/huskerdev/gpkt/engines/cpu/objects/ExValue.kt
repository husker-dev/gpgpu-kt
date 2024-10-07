package com.huskerdev.gpkt.engines.cpu.objects

import com.huskerdev.gpkt.ast.types.Type


open class ExValue(
    private var value: Any?
) {
    open fun set(newValue: Any?): ExValue{
        value = newValue
        return this
    }
    open fun get() = value

    fun castToType(type: Type?) = when{
        type == Type.FLOAT && value !is Float -> ExValue((value as Number).toFloat())
        type == Type.INT && value !is Double -> ExValue((value as Number).toInt())
        type == Type.BYTE && value !is Float -> ExValue((value as Number).toByte())
        else -> this
    }

}

class ExArrayAccessValue(
    private val array: Any,
    private var index: Int
): ExValue(null) {
    override fun set(newValue: Any?): ExValue{
        when (array) {
            is FloatArray -> array[index] = newValue as Float
            is IntArray -> array[index] = newValue as Int
            is ByteArray -> array[index] = newValue as Byte
            else -> throw UnsupportedOperationException("Can't set element in $array")
        }
        return this
    }
    override fun get(): Any = when (array) {
        is FloatArray -> array.getOrElse(index) { 0f }
        is IntArray -> array.getOrElse(index) { 0 }
        is ByteArray -> array.getOrElse(index) { 0.toByte() }
        else -> throw UnsupportedOperationException("Can't get element in $array")
    }
}

object BreakMarker: ExValue(null)
object ContinueMarker: ExValue(null)
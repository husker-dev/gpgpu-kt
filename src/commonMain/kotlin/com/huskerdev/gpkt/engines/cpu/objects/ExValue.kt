package com.huskerdev.gpkt.engines.cpu.objects


open class ExValue(
    private var value: Any?
) {
    open fun set(newValue: Any?): ExValue{
        value = newValue
        return this
    }
    open fun get() = value
}

class ExArrayAccessValue(
    private val array: Any,
    private var index: Int
): ExValue(null) {
    override fun set(newValue: Any?): ExValue{
        when (array) {
            is FloatArray -> array[index] = newValue as Float
            is IntArray -> array[index] = newValue as Int
            else -> throw UnsupportedOperationException("Can't set element in $array")
        }
        return this
    }
    override fun get(): Any = when (array) {
        is FloatArray -> array[index]
        is IntArray -> array[index]
        else -> throw UnsupportedOperationException("Can't get element in $array")
    }
}
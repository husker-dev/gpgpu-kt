package org.example.com.huskerdev.ta.sequence

import org.example.com.huskerdev.ta.engine.Engine
import org.example.com.huskerdev.ta.engine.gpu.EngineGPUImpl

var defaultEngine: Engine = EngineGPUImpl


// Pow

fun pow(seq1: NumSequence, seq2: NumSequence) =
    seq1.engine.pow(seq1, seq2)

fun pow(seq: NumSequence, power: Float) =
    seq.engine.pow(seq, power)

fun pow(seq: NumSequence, power: Int) =
    seq.engine.pow(seq, power.toFloat())

// Sqrt

fun sqrt(seq: NumSequence) =
    seq.engine.sqrt(seq)


abstract class NumSequence(
    val engine: Engine
) {

    class Root(
        engine: Engine,
        val values: FloatArray
    ): NumSequence(engine) {
        override fun get() = values
    }

    open fun get() =
        engine.calculate(this)

    infix fun shift(shift: Int) =
        engine.shift(this, shift)

    // Plus

    operator fun plus(other: NumSequence) =
        engine.plus(this, other)

    operator fun plus(num: Float) =
        engine.plus(this, num)

    operator fun plus(num: Int) =
        engine.plus(this, num.toFloat())

    // Minus

    operator fun minus(other: NumSequence) =
        engine.minus(this, other)

    operator fun minus(num: Float) =
        engine.minus(this, num)

    operator fun minus(num: Int) =
        engine.minus(this, num.toFloat())

    // Multiply

    operator fun times(other: NumSequence) =
        engine.multiply(this, other)

    operator fun times(num: Float) =
        engine.multiply(this, num)

    operator fun times(num: Int) =
        engine.multiply(this, num.toFloat())

    // Divide

    operator fun div(other: NumSequence) =
        engine.divide(this, other)

    operator fun div(num: Float) =
        engine.divide(this, num)

    operator fun div(num: Int) =
        engine.divide(this, num.toFloat())
}

fun FloatArray.toSequence(engine: Engine = defaultEngine) =
    NumSequence.Root(engine, this)

fun IntProgression.toSequence(engine: Engine = defaultEngine) =
    NumSequence.Root(engine, this.map { it.toFloat() }.toFloatArray())




package org.example.com.huskerdev.ta.engine.cpu

import org.example.com.huskerdev.ta.engine.Engine
import org.example.com.huskerdev.ta.sequence.NumSequence
import kotlin.math.pow

object EngineCPUImpl: Engine {
    override fun calculate(sequence: NumSequence): FloatArray {
        sequence as TransformSequence
        return FloatArray(sequence.roots.minOf { it.values.size }) {
            sequence[it]
        }
    }

    override fun shift(seq1: NumSequence, shift: Int): NumSequence {
        TODO("Not yet implemented")
    }

    override fun pow(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) { i ->
        seq1[i].pow(seq2[i])
    }

    override fun pow(seq1: NumSequence, num: Float) = TransformSequence(seq1) { i ->
        seq1[i].pow(num)
    }

    override fun sqrt(seq1: NumSequence) = TransformSequence(seq1) { i ->
        kotlin.math.sqrt(seq1[i])
    }

    override fun plus(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) { i ->
        seq1[i] + seq2[i]
    }

    override fun plus(seq1: NumSequence, num: Float) = TransformSequence(seq1) { i ->
        seq1[i] + num
    }

    override fun minus(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) { i ->
        seq1[i] - seq2[i]
    }

    override fun minus(seq1: NumSequence, num: Float) = TransformSequence(seq1) { i ->
        seq1[i] - num
    }

    override fun multiply(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) { i ->
        seq1[i] * seq2[i]
    }

    override fun multiply(seq1: NumSequence, num: Float) = TransformSequence(seq1) { i ->
        seq1[i] * num
    }

    override fun divide(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) { i ->
        seq1[i] / seq2[i]
    }

    override fun divide(seq1: NumSequence, num: Float) = TransformSequence(seq1) { i ->
        seq1[i] / num
    }

    private operator fun NumSequence.get(index: Int) = when (this) {
        is NumSequence.Root  -> this.values[index]
        is TransformSequence -> transform(index)
        else -> throw UnsupportedOperationException()
    }


    class TransformSequence(
        vararg sequences: NumSequence,
        val transform: (i: Int) -> Float
    ): NumSequence(this) {
        val roots: Array<Root>

        init {
            val roots = hashSetOf<Root>()
            sequences.forEach {
                if(it is Root)
                    roots += it
                else if(it is TransformSequence)
                    roots += it.roots
            }
            this.roots = roots.toTypedArray()
        }
    }
}
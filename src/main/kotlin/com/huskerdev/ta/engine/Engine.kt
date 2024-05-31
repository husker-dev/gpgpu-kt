package org.example.com.huskerdev.ta.engine

import org.example.com.huskerdev.ta.sequence.NumSequence


interface Engine {
    fun calculate(sequence: NumSequence): FloatArray

    fun shift(seq1: NumSequence, shift: Int): NumSequence

    fun pow(seq1: NumSequence, seq2: NumSequence): NumSequence
    fun pow(seq1: NumSequence, num: Float): NumSequence

    fun sqrt(seq1: NumSequence): NumSequence

    fun plus(seq1: NumSequence, seq2: NumSequence): NumSequence
    fun plus(seq1: NumSequence, num: Float): NumSequence

    fun minus(seq1: NumSequence, seq2: NumSequence): NumSequence
    fun minus(seq1: NumSequence, num: Float): NumSequence

    fun multiply(seq1: NumSequence, seq2: NumSequence): NumSequence
    fun multiply(seq1: NumSequence, num: Float): NumSequence

    fun divide(seq1: NumSequence, seq2: NumSequence): NumSequence
    fun divide(seq1: NumSequence, num: Float): NumSequence
}
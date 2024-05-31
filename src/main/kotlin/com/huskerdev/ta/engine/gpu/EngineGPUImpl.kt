package org.example.com.huskerdev.ta.engine.gpu

import org.example.com.huskerdev.ta.engine.Engine
import org.example.com.huskerdev.ta.sequence.NumSequence


object EngineGPUImpl: Engine {

    val opencl by lazy { OpenCL() }

    override fun calculate(sequence: NumSequence): FloatArray {
        sequence as TransformSequence

        val sequences = sequence.getSubSequences().apply { this.addFirst(sequence) }.distinct()
        val dictionary = Dictionary(sequences)

        val roots = sequence.roots.toList().toTypedArray()

        val inputs = sequences
            .filterIsInstance<NumSequence.Root>()
            .distinct()
            .map {
                "float ${dictionary[it]} = ${dictionary[it]}_[_i];"
            }
        val variables = sequences
            .filterIsInstance<TransformSequence>()
            .distinct()
            .joinToString(",", prefix = "float ", postfix = ";") {
                dictionary[it]
            }
        val operations = sequences
            .reversed()
            .filterIsInstance<TransformSequence>()
            .distinct()
            .map {
                "${dictionary[it]} = ${it.stringify(dictionary)};"
            }
        val tab = "                "

        val src = """
            __kernel void kernelMain(
                ${roots.joinToString(",\n$tab", postfix = ",") { "__global const float *${dictionary[it]}_" }}
                __global float *_res
            ) {
                int _i = get_global_id(0);
                ${inputs.joinToString("\n$tab")}
                $variables
                ${operations.joinToString("\n${tab}")}
                _res[_i] = ${dictionary[sequence]};
            }
        """.trimIndent()

        println(src)
        return opencl.run(src, roots.map { it.get() }.toTypedArray())
    }

    override fun shift(seq1: NumSequence, shift: Int): NumSequence {
        TODO("Not yet implemented")
    }

    override fun pow(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) {
        "pow(${it[seq1]}, ${it[seq2]})"
    }

    override fun pow(seq1: NumSequence, num: Float) = TransformSequence(seq1) {
        "pow(${it[seq1]}, ${num}f)"
    }

    override fun sqrt(seq1: NumSequence) = TransformSequence(seq1) {
        "sqrt(${it[seq1]})"
    }

    override fun plus(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) {
        "${it[seq1]} + ${it[seq2]}"
    }

    override fun plus(seq1: NumSequence, num: Float) = TransformSequence(seq1) {
        "${it[seq1]} + $num"
    }

    override fun minus(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) {
        "${it[seq1]} - ${it[seq2]}"
    }

    override fun minus(seq1: NumSequence, num: Float) = TransformSequence(seq1) {
        "${it[seq1]} - $num"
    }

    override fun multiply(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) {
        "${it[seq1]} * ${it[seq2]}"
    }

    override fun multiply(seq1: NumSequence, num: Float) = TransformSequence(seq1) {
        "${it[seq1]} * $num"
    }

    override fun divide(seq1: NumSequence, seq2: NumSequence) = TransformSequence(seq1, seq2) {
        "${it[seq1]} / ${it[seq2]}"
    }

    override fun divide(seq1: NumSequence, num: Float) = TransformSequence(seq1) {
        "${it[seq1]} / $num"
    }


    class TransformSequence(
        vararg val sequences: NumSequence,
        val stringify: (Dictionary) -> String
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

        fun getSubSequences(): MutableList<NumSequence> {
            val subSequences = arrayListOf(*sequences)
            sequences.forEach {
                if(it is TransformSequence)
                    subSequences += it.getSubSequences()
                else if(it is Root)
                    subSequences += it
            }
            return subSequences
        }
    }

    class Dictionary(
        sequences: List<NumSequence>
    ){
        private val alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray()
        val names = sequences
            .sortedBy { it is TransformSequence }
            .mapIndexed { index, root -> root to createUniqueName(index) }
            .toMap()

        operator fun get(key: NumSequence) = names[key]!!

        private fun createUniqueName(index: Int): String{
            val chars = arrayListOf<Char>()

            var curIndex = index + 1
            while(curIndex > alphabet.size) {
                chars.addFirst(alphabet[curIndex % alphabet.size - 1])
                curIndex /= alphabet.size
            }
            chars.addFirst(alphabet[curIndex - 1])

            return String(chars.toCharArray())
        }
    }
}
package com.huskerdev.gpkt.utils


class Dictionary(
    private var index: Int = 0
){
    private val alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray()

    fun copy() =
        Dictionary(index)

    fun nextWord(): String{
        val chars = arrayListOf<Char>()

        var curIndex = index++
        if(curIndex >= alphabet.size) {
            while(curIndex >= alphabet.size) {
                val i = curIndex % alphabet.size
                chars += alphabet[i]
                curIndex /= alphabet.size
            }
            chars += alphabet[curIndex % alphabet.size - 1]
        }else
            chars += alphabet[curIndex % alphabet.size]

        return chars.toCharArray().concatToString()
    }
}
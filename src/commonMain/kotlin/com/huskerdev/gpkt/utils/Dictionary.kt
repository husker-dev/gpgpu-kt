package com.huskerdev.gpkt.utils


class Dictionary {
    private val cachedNames = hashMapOf<Any, String>()

    operator fun get(key: Any) =
        cachedNames.computeIfAbsent(key) { createUniqueName(cachedNames.size) }
}

private val alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray()

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
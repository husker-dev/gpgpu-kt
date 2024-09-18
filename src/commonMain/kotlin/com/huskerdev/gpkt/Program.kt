package com.huskerdev.gpkt

abstract class Program {
    abstract fun execute(vararg mapping: Pair<String, Source>)
}
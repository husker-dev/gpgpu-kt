package com.huskerdev.gpkt

interface GPResource {
    val released: Boolean
    fun release()

    fun assertNotReleased() {
        if(released)
            throw UseOfReleasedObjectException()
    }
}

class UseOfReleasedObjectException: Exception("Can not use released object")
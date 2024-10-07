package com.huskerdev.gpkt.engines.opengl

import com.huskerdev.gpkt.MemoryUsage
import com.huskerdev.grapl.gl.GLContext
import com.huskerdev.grapl.gl.GLProfile
import org.lwjgl.opengl.GL
import org.lwjgl.opengl.GL30.glBindBufferBase
import org.lwjgl.opengl.GL43.*
import org.lwjgl.system.MemoryUtil


internal actual fun createGL(): OpenGL? = object: OpenGL{

    private val context = GLContext.create(majorVersion = 4, minorVersion = 3, coreProfile = GLProfile.CORE)
    override val name: String

    init {
        context.makeCurrent()
        GL.createCapabilities()

        name = glGetString(GL_RENDERER) ?: "unknown"
        GLContext.clear()
    }

    private inline fun useContextUnit(block: () -> Unit) {
        context.makeCurrent()
        block()
        GLContext.clear()
    }

    private inline fun <T> useContext(block: () -> T): T{
        context.makeCurrent()
        val a = block()
        GLContext.clear()
        return a
    }

    private fun MemoryUsage.toGL() = when(this){
        MemoryUsage.READ_ONLY -> GL_STATIC_DRAW
        MemoryUsage.WRITE_ONLY -> GL_DYNAMIC_READ
        MemoryUsage.READ_WRITE -> GL_DYNAMIC_COPY
    }

    private fun createSSBO(): Int{
        val ssbo = glGenBuffers()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        return ssbo
    }

    override fun deallocBuffer(ssbo: Int) = useContext {
        glDeleteBuffers(ssbo)
    }

    override fun deallocProgram(program: Int) = useContext {
        glDeleteProgram(program)
    }

    override fun alloc(size: Int, usage: MemoryUsage) = useContext {
        val ssbo = createSSBO()
        glBufferData(GL_SHADER_STORAGE_BUFFER, size.toLong(), usage.toGL())
        ssbo
    }

    override fun createProgram(source: String) = useContext {
        val compute = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(compute, source)
        glCompileShader(compute)
        if(glGetShaderi(compute, GL_COMPILE_STATUS) != GL_TRUE)
            throw Exception("Failed to compile GL shader:\n${glGetShaderInfoLog(compute)}")

        val program = glCreateProgram()
        glAttachShader(program, compute)
        glLinkProgram(program)
        if(glGetProgrami(program, GL_LINK_STATUS) != GL_TRUE)
            throw Exception("Failed to link GL program:\n${glGetProgramInfoLog(program)}")
        glDeleteShader(compute)
        program
    }

    override fun useProgram(program: Int) = useContext {
        glUseProgram(program)
    }

    override fun launchProgram(instances: Int) = useContext {
        glDispatchCompute(instances, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT or GL_BUFFER_UPDATE_BARRIER_BIT)
    }

    override fun setBufferIndex(index: Int, ssbo: Int) = useContext {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo)
    }

    override fun setUniform1f(index: Int, value: Float) = useContext {
        glUniform1f(index, value)
    }

    override fun setUniform1i(index: Int, value: Int) = useContext {
        glUniform1i(index, value)
    }

    override fun setUniform1b(index: Int, value: Byte) = useContext {
        //glUniform1(index, value)
    }

    override fun wrapFloat(array: FloatArray, usage: MemoryUsage) = useContext {
        val ssbo = createSSBO()
        val buffer = MemoryUtil.memAllocFloat(array.size).put(array).flip()
        glBufferData(GL_SHADER_STORAGE_BUFFER, buffer, usage.toGL())
        MemoryUtil.memFree(buffer)
        ssbo
    }

    override fun wrapDouble(array: DoubleArray, usage: MemoryUsage) = useContext {
        val ssbo = createSSBO()
        val buffer = MemoryUtil.memAllocDouble(array.size).put(array).flip()
        glBufferData(GL_SHADER_STORAGE_BUFFER, buffer, usage.toGL())
        MemoryUtil.memFree(buffer)
        ssbo
    }

    override fun wrapInt(array: IntArray, usage: MemoryUsage) = useContext {
        val ssbo = createSSBO()
        val buffer = MemoryUtil.memAllocInt(array.size).put(array).flip()
        glBufferData(GL_SHADER_STORAGE_BUFFER, buffer, usage.toGL())
        MemoryUtil.memFree(buffer)
        ssbo
    }

    override fun wrapByte(array: ByteArray, usage: MemoryUsage) = useContext {
        val ssbo = createSSBO()
        val buffer = MemoryUtil.memAlloc(array.size).put(array).flip()
        glBufferData(GL_SHADER_STORAGE_BUFFER, buffer, usage.toGL())
        MemoryUtil.memFree(buffer)
        ssbo
    }

    override fun readFloat(ssbo: Int, length: Int, offset: Int) = useContext {
        val buffer = MemoryUtil.memAllocFloat(length)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset.toLong(), buffer)
        val array = FloatArray(length)
        buffer.get(array)
        MemoryUtil.memFree(buffer)
        array
    }

    override fun readInt(ssbo: Int, length: Int, offset: Int)= useContext {
        val buffer = MemoryUtil.memAllocInt(length)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset.toLong(), buffer)
        val array = IntArray(length)
        buffer.get(array)
        MemoryUtil.memFree(buffer)
        array
    }

    override fun readByte(ssbo: Int, length: Int, offset: Int)= useContext {
        val buffer = MemoryUtil.memAlloc(length)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, offset.toLong(), buffer)
        val array = ByteArray(length)
        buffer.get(array)
        MemoryUtil.memFree(buffer)
        array
    }

    override fun writeFloat(ssbo: Int, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int) = useContextUnit {
        val buffer = MemoryUtil.memAllocFloat(length).put(src, srcOffset, length).flip()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, dstOffset.toLong(), buffer)
        MemoryUtil.memFree(buffer)
    }

    override fun writeInt(ssbo: Int, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int) = useContextUnit {
        val buffer = MemoryUtil.memAllocInt(length).put(src, srcOffset, length).flip()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, dstOffset.toLong(), buffer)
        MemoryUtil.memFree(buffer)
    }

    override fun writeByte(ssbo: Int, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int) = useContextUnit {
        val buffer = MemoryUtil.memAlloc(length).put(src, srcOffset, length).flip()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, dstOffset.toLong(), buffer)
        MemoryUtil.memFree(buffer)
    }

}
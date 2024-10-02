package com.huskerdev.gpkt.engines.webgpu


import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.await
import com.huskerdev.gpkt.utils.jsObject
import kotlinx.browser.window
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Uint8Array
import kotlin.js.Promise

class WebGPU(
    private val device: dynamic,
    private var commandEncoder: dynamic,
    val name: String
) {

    companion object {
        val supported = window.navigator.asDynamic().gpu != null

        suspend fun create(): WebGPU {
            val navigator = window.navigator.asDynamic()
            val adapter = (navigator.gpu.requestAdapter() as Promise<dynamic>).await()
            val device = (adapter.requestDevice() as Promise<dynamic>).await()
            val commandEncoder = device.createCommandEncoder()

            val info = adapter.info
            val name = if (info != null)
                "${info.vendor} (${info.architecture})"
            else "unnamed"

            return WebGPU(device, commandEncoder, name)
        }
    }

    fun alloc(size: Int): dynamic{
        println("alloc(${size})")

        return device.createBuffer(jsObject {
            this.size = size
            usage = js("GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST")
        })
    }

    fun alloc(byteArray: ArrayBuffer): dynamic{
        println("alloc([${byteArray.byteLength}])")

        val buffer = device.createBuffer(jsObject {
            mappedAtCreation = true
            size = byteArray.byteLength
            usage = js("GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST")
        })

        val arrayBuffer = Uint8Array(buffer.getMappedRange() as ArrayBuffer)
        arrayBuffer.set(Uint8Array(byteArray))
        buffer.unmap()
        return buffer
    }

    fun allocWrite(byteArray: ArrayBuffer): dynamic{
        println("allocWrite([${byteArray.byteLength}])")

        val writeBuffer = device.createBuffer(jsObject {
            mappedAtCreation = true
            size = byteArray.byteLength
            usage = js("GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC")
        })

        // Fill it
        val arrayBuffer = Uint8Array(writeBuffer.getMappedRange() as ArrayBuffer)
        arrayBuffer.set(Uint8Array(byteArray))
        writeBuffer.unmap()
        return writeBuffer
    }

    fun allocRead(size: Int): dynamic{
        println("allocRead([$size])")

        return device.createBuffer(jsObject {
            this.size = size
            usage = js("GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST")
        })
    }

    fun dealloc(buffer: dynamic){
        buffer.destroy()
    }

    fun copyBufferToBuffer(src: dynamic, dst: dynamic, srcOffset: Int, dstOffset: Int, size: Int){
        commandEncoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size)
        flush()
    }

    private fun flush(){
        device.queue.submit(arrayOf(commandEncoder.finish()))
        commandEncoder = device.createCommandEncoder()
    }

    fun createGroupLayout(bufferFields: List<Field>): dynamic{
        return device.createBindGroupLayout(jsObject {
            entries = js("[]")
            bufferFields.forEachIndexed { i, buffer ->
                val o = js("{}")
                o.binding = i
                o.visibility = js("GPUShaderStage.COMPUTE")
                o.buffer = js("{}")
                o.buffer.type = if(Modifiers.READONLY in buffer.modifiers)
                    "read-only-storage" else "storage"
                entries.push(o)
            }
        })
    }

    fun createShaderModule(source: String): dynamic{
        return device.createShaderModule(jsObject {
            code = source
        })
    }

    fun createPipeline(shaderModule: dynamic, groupLayout: dynamic, mainMethod: String): dynamic{
        return device.createComputePipeline(jsObject {
            layout = device.createPipelineLayout(jsObject{
                bindGroupLayouts = js("[]")
                bindGroupLayouts.push(groupLayout)
            })
            compute = jsObject {
                module = shaderModule
                entryPoint = mainMethod
            }
        })
    }

    fun execute(groupLayout: dynamic, pipeline: dynamic, buffers: List<WebGPUMemoryPointer<*>>){
        val bindGroup = device.createBindGroup(jsObject {
            layout = groupLayout
            entries = js("[]")
            buffers.forEachIndexed { i, buffer ->
                val o = js("{}")
                o.binding = i
                o.resource = js("{}")
                o.resource.buffer = buffer.gpuBuffer
                entries.push(o)
            }
        })

        val passEncoder = commandEncoder.beginComputePass()
        passEncoder.setPipeline(pipeline)
        passEncoder.setBindGroup(0, bindGroup)
        passEncoder.dispatchWorkgroups(128, 128)
        passEncoder.end()

        flush()
    }
}


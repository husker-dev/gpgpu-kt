package com.huskerdev.gpkt.apis.webgpu


import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.await
import com.huskerdev.gpkt.utils.jsObject
import kotlinx.browser.window
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Uint8Array
import kotlin.js.Promise

class WebGPU{

    companion object {
        val supported = window.navigator.asDynamic().gpu != null
    }

    suspend fun requestAdapter(): dynamic{
        val navigator = window.navigator.asDynamic()
        return (navigator.gpu.requestAdapter() as Promise<dynamic>).await()
    }

    suspend fun requestDevice(adapter: dynamic): dynamic{
        return (adapter.requestDevice() as Promise<dynamic>).await()
    }

    fun getAdapterName(adapter: dynamic): String{
        val info = adapter.info
        return if (info != null)
            "${info.vendor} (${info.architecture})"
        else "unnamed"
    }

    fun createCommandEncoder(device: dynamic) =
        device.createCommandEncoder()

    fun dispose(device: dynamic){
        device.destroy()
    }

    fun alloc(device: dynamic, size: Int): dynamic{
        return device.createBuffer(jsObject {
            this.size = size
            usage = js("GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST")
        })
    }

    fun alloc(device: dynamic, byteArray: ArrayBuffer): dynamic{
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

    fun allocWrite(device: dynamic, byteArray: ArrayBuffer): dynamic{
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

    fun allocRead(device: dynamic, size: Int): dynamic{
        return device.createBuffer(jsObject {
            this.size = size
            usage = js("GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST")
        })
    }

    fun dealloc(buffer: dynamic){
        buffer.destroy()
    }

    fun copyBufferToBuffer(commandEncoder: dynamic, src: dynamic, dst: dynamic, srcOffset: Int, dstOffset: Int, size: Int){
        commandEncoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size)
    }

    fun flush(device: dynamic, commandEncoder: dynamic): dynamic{
        device.queue.submit(arrayOf(commandEncoder.finish()))
        return device.createCommandEncoder()
    }

    fun createGroupLayout(device: dynamic, bufferFields: List<Field>): dynamic{
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

    fun createShaderModule(device: dynamic, source: String): dynamic{
        return device.createShaderModule(jsObject {
            code = source
        })
    }

    fun createPipeline(device: dynamic, shaderModule: dynamic, groupLayout: dynamic, mainMethod: String): dynamic{
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

    fun execute(device: dynamic, commandEncoder: dynamic, groupLayout: dynamic, pipeline: dynamic, buffers: List<WebGPUMemoryPointer<*>>){
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
    }
}


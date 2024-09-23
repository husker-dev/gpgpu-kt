package com.huskerdev.gpkt.engines.jdk

import java.io.ByteArrayOutputStream
import java.io.FilterOutputStream
import java.io.StringWriter
import java.net.URI
import java.net.URL
import java.net.URLClassLoader
import java.nio.CharBuffer
import javax.tools.*


class ClassCompiler {

    companion object {
        fun compileClass(source: String, classPath: String): Class<*>{
            val compiler = ToolProvider.getSystemJavaCompiler()
            val stdManager = compiler.getStandardFileManager(null, null, null)

            MemoryJavaFileManager(stdManager).use { manager ->
                val className = classPath.split(".").last()
                val file = MemoryJavaFileManager.MemoryInputJavaFileObject("$className.java", source)

                val output = StringWriter()
                val task = compiler.getTask(output, manager, null, null, null, listOf(file))

                val result = task.call()
                if (result == null || !result)
                    throw RuntimeException(output.buffer.toString())

                val classLoader = MemoryClassLoader(manager.classBytes)
                return classLoader.loadClass(classPath)
            }
        }
    }

    class MemoryJavaFileManager(fileManager: JavaFileManager?) : ForwardingJavaFileManager<JavaFileManager>(fileManager) {
        lateinit var classBytes: ByteArray

        override fun flush() = Unit
        override fun close() = Unit

        override fun getJavaFileForOutput(
            location: JavaFileManager.Location?, className: String, kind: JavaFileObject.Kind,
            sibling: FileObject?
        ): JavaFileObject =
            if (kind === JavaFileObject.Kind.CLASS) MemoryOutputJavaFileObject(className)
            else super.getJavaFileForOutput(location, className, kind, sibling)

        class MemoryInputJavaFileObject(name: String, private val code: String): SimpleJavaFileObject(
            URI.create("string:///$name"),
            JavaFileObject.Kind.SOURCE
        ) {
            override fun getCharContent(ignoreEncodingErrors: Boolean): CharBuffer =
                CharBuffer.wrap(code)
        }

        inner class MemoryOutputJavaFileObject(name: String): SimpleJavaFileObject(
            URI.create("string:///$name"),
            JavaFileObject.Kind.CLASS
        ) {
            override fun openOutputStream() = object : FilterOutputStream(ByteArrayOutputStream()) {
                override fun close() {
                    out.close()
                    classBytes = (out as ByteArrayOutputStream).toByteArray()
                }
            }
        }
    }

    class MemoryClassLoader(
        private val classBytes: ByteArray
    ) : URLClassLoader(
        arrayOfNulls<URL>(0),
        MemoryClassLoader::class.java.classLoader
    ) {
        override fun findClass(name: String): Class<*> =
            defineClass(name, classBytes, 0, classBytes.size)
    }
}
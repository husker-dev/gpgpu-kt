# gpgpu-kt
Cross-platform general-purpose computing Kotlin Multiplatform library

### Available executions:
 - **Common**
   - Interpreter
 - **JVM**
   - OpenCL
   - CUDA
   - javac
 - **JS, Wasm**
   - WebGPU
   - JS

### Planned executions:
 - **iOS**
   - Metal
 - **Android**
   - OpenCL
   - Vulkan

### TODO List:
  - [x] Add standard functions (pow, sqrt, etc)
  - [x] Add argument `from` to execute()
  - [x] Add single value updating in memory objects
  - [x] Add JS support
  - [x] Add WebGPU support (Beta)
  - [ ] Add array creation support
  - [ ] Add Vulkan support
  - [ ] Add Metal support

## Simple usage 
1. Create GPDevice
```kotlin
val device = GPDevice.create()
```

2. Compile a program
```kotlin
val program = device.compile("""
    extern float[] arr1, arr2, result;
    extern float multiplier;
    
    void main(const int i){
        result[i] = arr1[i] + arr2[i] * multiplier;
    }
""".trimIndent())
```

3. Allocate buffers on GPU
```kotlin
val arr1 = device.allocFloat(exampleArray())
val arr2 = device.allocFloat(exampleArray())
val result = engine.allocFloat(arr1.length)

fun exampleArray() = FloatArray(1_000_000) { it.toFloat() }
```

4. Execute
```kotlin
program.execute(
    instances = 1_000_000,
    "arr1" to arr1,
    "arr2" to arr2,
    "result" to result,
    "multiplier" to 2
)
```

5. Read changed buffers
```kotlin
result.read()
```

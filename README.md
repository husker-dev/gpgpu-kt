# gpgpu-kt
Cross-platform general-purpose computing Kotlin Multiplatform library

### Available accelerations:
|         | OpenCL             | CUDA               | Vulkan | Metal                | WebGPU    | JS                 | Bytecode           |
|---------|--------------------|--------------------|--------|----------------------|-----------|--------------------|--------------------|
| jvm     | :white_check_mark: | :white_check_mark: | :x:    |  :x:                 |           |                    | :white_check_mark: |
| js      |                    |                    |        |                      |:warning:* | :white_check_mark: |                    |
| wasm    |                    |                    |        |                      |    :x:    |         :x:        |                    |
| ios     |                    |                    |        |  :white_check_mark:  |           |                    |                    |
| android | :white_check_mark: |                    | :x:    |                      |           |                    |  :x:               |
| macos   | :x:                |                    |        |  :white_check_mark:  |           |                    |                    |
| windows | :x:                | :white_check_mark: | :x:    |                      |           |                    |                    |
| linux   | :x:                | :white_check_mark: | :x:    |                      |           |                    |                    |

- :white_check_mark: - Fully implemented
- :warning: - Partially working
- :x: - Not implemented, but planned

- \* Doesn't support loops and Byte 

### TODO List:
  - [ ] Add `sizeof()` function
  - [ ] Add array creation support
  - [ ] Add Vulkan support

## Simple example 
1. Create GPDevice
```kotlin
val device = GPDevice.create()
```

2. Compile a program
```kotlin
val program = device.compile("""
    extern readonly float[] arr1, arr2;
    extern float[] result;
    extern float multiplier;
    
    void main(const int i){
        result[i] = arr1[i] + arr2[i] * multiplier;
    }
""".trimIndent())
```

3. Allocate buffers in GPU memory
```kotlin
val arr1 = device.wrapFloats(exampleArray())
val arr2 = device.wrapFloats(exampleArray())
val result = engine.allocFloats(arr1.length)

fun exampleArray() = FloatArray(1_000_000) { it.toFloat() }
```

4. Execute the program with arguments
```kotlin
program.execute(
    instances = 1_000_000,
    "arr1" to arr1,
    "arr2" to arr2,
    "result" to result,
    "multiplier" to 2
)
```

5. Read the result from buffer
```kotlin
val modifiedResultArray = result.read()
```

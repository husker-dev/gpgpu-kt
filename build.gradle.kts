plugins {
    id("org.jetbrains.kotlin.multiplatform").version("2.0.20")
    id("maven-publish")
}

group = "com.huskerdev"
version = "1.0"

repositories {
    mavenCentral()
}

kotlin {
    jvm()
    js()

    sourceSets {
        jvmMain {
            dependencies {
                // JOCL (OpenCL)
                implementation("org.jocl:jocl:2.0.4")

                // JCuda (Cuda)
                implementation("org.jcuda:jcuda:12.0.0"){
                    isTransitive = false
                }
                implementation("org.jcuda:jcuda-natives:12.0.0:windows-x86_64")
                implementation("org.jcuda:jcuda-natives:12.0.0:linux-x86_64")
            }
        }
    }
}
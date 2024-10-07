@file:OptIn(ExperimentalKotlinGradlePluginApi::class)
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJsCompile


plugins {
    id("org.jetbrains.kotlin.multiplatform") version "2.0.20"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.11"

    id("com.android.library") version "8.6.1"

    id("maven-publish")
}

group = "com.huskerdev"
version = "1.0"

repositories {
    google()
    mavenCentral()
}

kotlin {
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
        compilations.create("benchmark") {
            associateWith(this@jvm.compilations.getByName("main"))
        }
    }
    js {
        browser()
        binaries.executable()
    }
    macosArm64()
    macosX64()
    iosX64()
    iosArm64()

    androidTarget {
        publishLibraryVariants("release")
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_1_8)
        }
    }

    sourceSets {
        val macosMain by creating {
            macosX64Main.get().dependsOn(this)
            macosArm64Main.get().dependsOn(this)
        }

        val iosMain by creating {
            iosX64Main.get().dependsOn(this)
            iosArm64Main.get().dependsOn(this)
        }

        val commonOpenGL by creating {
            dependsOn(commonMain.get())

            jvmMain.get().dependsOn(this)
        }

        val commonOpenCL by creating {
            dependsOn(commonMain.get())

            jvmMain.get().dependsOn(this)
            androidMain.get().dependsOn(this)
        }

        val commonMetal by creating {
            dependsOn(commonMain.get())

            //jvmMain.get().dependsOn(this)
        }

        val commonMetalNative by creating {
            dependsOn(commonMetal)

            macosMain.dependsOn(this)
            iosMain.dependsOn(this)
        }

        commonTest.dependencies {
            implementation(kotlin("test"))
        }

        jvmMain {
            dependencies {
                // JOCL (OpenCL)
                api("org.jocl:jocl:2.0.4")

                // JCuda (Cuda)
                api("org.jcuda:jcuda:12.0.0") {
                    isTransitive = false
                }
                api("org.jcuda:jcuda-natives:12.0.0:windows-x86_64")
                api("org.jcuda:jcuda-natives:12.0.0:linux-x86_64")

                // OpenGL
                api("com.huskerdev:grapl-gl:2.2.0")
                implementation(project.dependencies.platform("org.lwjgl:lwjgl-bom:3.3.4"))
                api("org.lwjgl:lwjgl")
                api("org.lwjgl:lwjgl-opengl")
                api(fileTree("natives"))
            }
        }

        val jvmBenchmark by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.11")
            }
        }
    }
}

android {
    namespace = "com.huskerdev.gpkt"
    compileSdk = 34
    ndkVersion = "19.2.5345600"

    defaultConfig {
        minSdk = 19
    }

    externalNativeBuild {
        cmake {
            path("src/androidMain/cpp/CMakeLists.txt")
            version = "3.30.4"
        }
    }
}


benchmark {
    targets {
        register("jvmBenchmark")
    }
}

tasks.withType(KotlinJsCompile::class.java).configureEach {
    compilerOptions {
        target = "es2015"
    }
}
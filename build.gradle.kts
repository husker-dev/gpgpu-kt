plugins {
    id("org.jetbrains.kotlin.multiplatform").version("2.0.20")
}

group = "com.huskerdev"
version = "1.0"

repositories {
    mavenCentral()
}

kotlin {
    jvm()

    sourceSets {
        jvmMain {
            dependencies {
                implementation("org.jocl:jocl:2.0.4")
            }
        }
    }
}
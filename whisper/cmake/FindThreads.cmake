# This file just marks the threads library as found for whisper's build system.
# It doesn't actually provide it, but I stripped out all uses of it anyway.

set(Threads_FOUND TRUE)

add_library(Threads::Threads INTERFACE IMPORTED)
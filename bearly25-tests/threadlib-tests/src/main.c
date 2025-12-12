#include "main.h"
#include "chip_config.h"
#include "tests.h"
#include "hthread.h"
#include "riscv.h"
#include <stdio.h>

void app_init() {
    printf("=== Threading test app init ===\n");
    hthread_init();
}

void app_main() {
    uint64_t mhartid = READ_CSR("mhartid");
    printf("Hello from app_main on hart %ld\n", (long)mhartid);

    if (mhartid == 0) {
        run_all_tests();
    }

    printf("app_main completed on hart %ld\n", (long)mhartid);
}

int main(void) {
    app_init();
    app_main();
    return 0;
}

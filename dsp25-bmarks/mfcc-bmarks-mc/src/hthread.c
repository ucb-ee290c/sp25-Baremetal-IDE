#include "hthread.h"

typedef struct {
  volatile uint32_t flags[N_HARTS];
  void *(*start_routines[N_HARTS])(void *);
  void *args[N_HARTS];
} hthread_info_t;

static volatile hthread_info_t hthread_dat;

void hthread_issue(uint32_t hartid, void *(* start_routine)(void *), void *arg) {
    hthread_dat.start_routines[hartid] = start_routine;
    hthread_dat.args[hartid] = arg;
    asm volatile("fence rw, rw" ::: "memory");
    hthread_dat.flags[hartid] = 1;
    CLINT->MSIP[hartid] = 1;
}

void hthread_join(uint32_t hartid) {
    while(hthread_dat.flags[hartid] == 1) {
        asm volatile("nop");
    }
}

void __main(void) {
  uint64_t mhartid = READ_CSR("mhartid");
  /* Enable machine software interrupt in mie so WFI returns on MSIP.
   * mstatus.MIE remains 0 so no actual trap is taken. */
  SET_CSR_BITS("mie", 1 << 3);
  while (1) {
    while (hthread_dat.flags[mhartid] == 0) {
      asm volatile("wfi");
    }
    CLINT->MSIP[mhartid] = 0;
    hthread_dat.start_routines[mhartid](hthread_dat.args[mhartid]);
    asm volatile("fence rw, rw" ::: "memory");
    hthread_dat.flags[mhartid] = 0;
  }
}

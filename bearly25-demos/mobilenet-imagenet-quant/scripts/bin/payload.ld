OUTPUT_FORMAT("elf64-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS {
  . = 0x81c35000;
  .data : { *(.data) }
  /DISCARD/ : { *(*) }
}

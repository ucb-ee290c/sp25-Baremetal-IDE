PREFIX = riscv64-unknown-elf-

CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OD = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

PORT = 3333
TYPE = Release

.PHONY: build
build:
	cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=$(TYPE) -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -DCHIP=$(CHIP)
	cmake --build ./build/ --target $(TARGET)

.PHONY: ocd
ocd:
	openocd -f ./platform/$(CHIP)/$(CHIP).cfg

.PHONY: ocd-run
ocd-run:
	openocd -f ./platform/$(CHIP)/$(CHIP).cfg -c "reset run" -c "halt" -c "load_image $(BINARY)" -c "resume 0x80000000"

.PHONY: gdb
gdb:
	$(DG) $(BINARY) --eval-command="target extended-remote localhost:$(PORT)"
	# --eval-command="monitor reset"

.PHONY: clean
clean:
	rm -rf build

.PHONY: dump
dump:
	$(OD) -D  $(BINARY) > $(BINARY).dump

.PHONY: checktsi
checktsi:
	uart_tsi +tty=$(TTY) +baudrate=921600 +no_hart0_msip +init_write=0x80001000:0xb0bacafe +init_read=0x80001000 none

.PHONY: tsi-run
tsi-run:
	uart_tsi +tty=$(TTY) +baudrate=921600 $(BINARY)

.PHONY: vcs-run
vcs-run:
	echo "Running VCS tests within $(CY_DIR)"
	(cd $(CY_DIR)/sims/vcs && make run-binary CONFIG=$(CONFIG) BINARY=$(BINARY))

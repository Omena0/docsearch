# Bootloader Documentation

# Bootloader

The bootloader is responsible for loading the kernel.

## Stages

The bootloader has multiple stages.

### Stage 1

Stage 1 is the MBR/BIOS boot sector.

Loads stage 2 from disk.

### Stage 2

Stage 2 loads the kernel and initrd.

## Configuration

Bootloader configuration is stored in config files.

### GRUB

GRUB is the default bootloader.

#### Menu Entries

Each menu entry defines a kernel to boot.

#### Parameters

Kernel parameters are passed via command line.

## UEFI

UEFI booting is also supported.

### Boot Entries

UEFI boot entries are stored in NVRAM.
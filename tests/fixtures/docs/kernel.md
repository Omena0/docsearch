# Kernel Documentation

# Overview

This documentation describes the kernel architecture, processes, memory management, and system calls.

## Architecture

The kernel is a monolithic design with modular components.

### Boot Process

The bootloader loads the kernel into memory and transfers control.

### Initialization

After boot, the kernel initializes subsystems in order.

# Processes

Processes are the fundamental unit of execution.

## Creation

Processes are created via fork and exec.

### fork

The fork system call creates a child process by duplicating the parent.

The child receives a copy of the parent's address space.

```c
pid_t fork(void);
```

Returns 0 in child, child PID in parent.

### exec

The exec family replaces the current process image.

```c
int execve(const char *path, char *const argv[], char *const envp[]);
```

Loads a new program into the process.

## Scheduling

The scheduler uses a CFS-like algorithm.

### Time Slices

Each process gets a time slice based on priority.

## Signals

Signals provide asynchronous notification.

### Signal Handling

Processes can register handlers for signals.

# Memory Management

Virtual memory is managed via page tables.

## Page Cache

The page cache stores file data in memory.

Pages are evicted under memory pressure.

## Swapping

Anonymous pages can be swapped to disk.

# System Calls

System calls are the kernel-userspace interface.

## Process Management

fork, exec, wait, exit.

## File Operations

open, read, write, close.

## Memory

mmap, munmap, mprotect.
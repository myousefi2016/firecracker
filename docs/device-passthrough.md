# PCIe device passthrough (VFIO)

Firecracker can assign a physical PCIe device on the host directly to a microVM
using the Linux [VFIO](https://docs.kernel.org/driver-api/vfio.html) framework.
This is commonly used to give a guest direct, low overhead access to an
accelerator such as a GPU, but it works for any PCIe function that the host can
bind to the `vfio-pci` driver (NICs, NVMe drives, other accelerators, ...).

With passthrough the guest drives the real hardware: its BARs are mapped
straight into the guest address space (no VM exit on access) and the device's
MSI-X interrupts are delivered to the guest through KVM irqfds. The IOMMU
isolates the device so it can only DMA into memory that Firecracker explicitly
mapped for the guest.

Passthrough is built on top of the PCIe transport, so it requires the
`--enable-pci` flag (see [getting started](getting-started.md)).

## Requirements

- An IOMMU enabled on the host (Intel VT-d or AMD-Vi on x86_64, SMMU on
  aarch64). On x86_64 this usually means booting the host kernel with
  `intel_iommu=on` (or `amd_iommu=on`) and `iommu=pt`.
- The host kernel built with VFIO support: `CONFIG_VFIO`, `CONFIG_VFIO_PCI` and
  `CONFIG_VFIO_IOMMU_TYPE1`.
- The guest kernel built with PCI and MSI-X support (see
  [kernel policy](kernel-policy.md)).
- The assigned device must be bound to the `vfio-pci` driver on the host before
  it is handed to Firecracker.

## Preparing a device on the host

First find the device's PCI address and its IOMMU group:

```bash
lspci -nn
# e.g. 01:00.0 3D controller [0302]: ... [10de:1eb8] (the GPU)

readlink -f /sys/bus/pci/devices/0000:01:00.0/iommu_group
# e.g. /sys/kernel/iommu_groups/15
```

All functions in the same IOMMU group must be assigned together. Unbind each
function from its host driver and bind it to `vfio-pci`:

```bash
echo 0000:01:00.0 > /sys/bus/pci/devices/0000:01:00.0/driver/unbind
echo 10de 1eb8 > /sys/bus/pci/drivers/vfio-pci/new_id
```

After this, `/dev/vfio/15` (the group node) and `/dev/vfio/vfio` (the container)
exist and are owned by root.

## Assigning the device to a microVM

Passthrough devices can only be added before the microVM is started, and only
when PCI is enabled. The device is referenced by its host sysfs path.

Using the API:

```bash
curl --unix-socket "${API_SOCKET}" -i \
    -X PUT 'http://localhost/vfio/gpu0' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "gpu0",
        "path": "/sys/bus/pci/devices/0000:01:00.0"
    }'
```

Using a JSON configuration file (`--config-file`), add a `vfio` array:

```json
{
  "vfio": [
    {
      "id": "gpu0",
      "path": "/sys/bus/pci/devices/0000:01:00.0"
    }
  ]
}
```

Inside the guest the device then shows up under `lspci`, bound to whatever guest
driver matches it (for example the NVIDIA driver for an NVIDIA GPU). Because the
device is the real hardware, no special guest configuration beyond installing
the appropriate driver is required.

## How it works

- **Configuration space** is passed through to the physical device, except the
  Base Address Registers, which are virtualized so the guest sees the guest
  physical addresses Firecracker assigned rather than the host addresses.
- **BARs** are mapped directly into the guest as KVM memory regions for
  exit-less access. The MSI-X table and pending bit array are trapped so that
  interrupt delivery can be virtualized.
- **MSI-X** interrupts raised by the device are routed by VFIO into KVM irqfds,
  which inject them into the guest with no userspace involvement.
- **DMA**: the whole guest physical address space is mapped into the device's
  IOMMU domain, so the device can DMA to and from guest RAM, and only guest RAM.

## Limitations

- Passthrough requires `--enable-pci`.
- A microVM that has a passthrough device attached cannot be snapshotted: the
  internal state of a physical device lives in the hardware and cannot be
  captured, so [snapshotting](snapshotting/snapshot-support.md) is rejected
  while such a device is attached.
- Passthrough devices can only be attached before boot; they cannot be
  hot-plugged or hot-unplugged.
- Only memory BARs are supported (legacy IO BARs are not), which is what modern
  PCIe devices use.
- When running under the [jailer](jailer.md), the relevant `/dev/vfio` nodes
  must be made available inside the chroot.

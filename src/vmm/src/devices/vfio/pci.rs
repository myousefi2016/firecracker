// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! PCI emulation for a VFIO assigned device.
//!
//! [`VfioPciDevice`] turns a raw [`VfioPciResources`] (a physical PCI function exposed through
//! VFIO) into a device the guest can probe and drive as if it were attached to the guest's
//! emulated PCI bus. It implements [`PciDevice`] so it can be plugged into the same PCI bus and
//! MMIO bus machinery used by virtio-pci devices.
//!
//! The implementation mirrors the well established model used by Cloud Hypervisor and QEMU:
//!
//! * **Configuration space** is, with two exceptions, passed straight through to the physical
//!   device via the VFIO config region. The exceptions are the Base Address Registers (BARs) and
//!   the expansion ROM BAR, which are virtualized: the guest sees the *guest* physical addresses
//!   Firecracker assigned, not the host addresses the host kernel programmed, and writes to them
//!   are trapped locally instead of reprogramming the real device.
//! * **BARs** are enumerated from the VFIO region info. Memory BARs that the kernel allows to be
//!   `mmap`'d are mapped straight into the guest address space as KVM memory regions, so the guest
//!   accesses the hardware with no VM exit. The MSI-X table and PBA live in a trapped hole so that
//!   Firecracker can virtualize interrupt delivery.
//! * **MSI-X** is virtualized through [`MsixConfig`]: guest writes to the MSI-X table are trapped
//!   and translated into KVM GSI routes / irqfds, while the physical device's MSI-X interrupts are
//!   routed to those same irqfds through VFIO, giving a direct, exit-less interrupt path.

use std::os::fd::{AsRawFd, RawFd};
use std::sync::{Arc, Barrier, Mutex};

use vfio_bindings::bindings::vfio::{
    VFIO_PCI_BAR0_REGION_INDEX, VFIO_PCI_CONFIG_REGION_INDEX, VFIO_PCI_MSIX_IRQ_INDEX,
    VFIO_REGION_INFO_FLAG_MMAP, VFIO_REGION_INFO_FLAG_READ, VFIO_REGION_INFO_FLAG_WRITE,
};
use vfio_ioctls::{VfioDevice, VfioRegionInfoCap};
use vmm_sys_util::eventfd::EventFd;

use crate::devices::vfio::VfioPciResources;
use crate::logger::{debug, error, warn};
use crate::pci::PciDevice;
use crate::pci::PciSBDF;
use crate::pci::configuration::{BAR0_REG_IDX, BarPrefetchable, Bars, NUM_BAR_REGS};
use crate::pci::msix::MsixConfig;
use crate::vstate::bus::BusDevice;
use crate::vstate::interrupts::MsixVectorGroup;
use crate::vstate::vm::KvmVm;

/// Size in bytes of a single MSI-X table entry, as defined by the PCI Local Bus specification.
const MSIX_TABLE_ENTRY_SIZE: u64 = 16;
/// PCI configuration space register index of the first BAR (`0x10 / 4`).
const PCI_CONFIG_BAR0_INDEX: u16 = BAR0_REG_IDX;
/// PCI configuration space register index of the expansion ROM BAR (`0x30 / 4`).
const PCI_CONFIG_ROM_BAR_INDEX: u16 = 12;
/// Byte offset of the first BAR in PCI configuration space.
const PCI_CONFIG_BAR_OFFSET: u32 = 0x10;
/// Status register bit indicating the presence of a capability list.
const PCI_CONFIG_STATUS_CAPABILITIES_LIST: u32 = 0x0010_0000;
/// Offset of the capabilities list pointer in PCI configuration space.
const PCI_CONFIG_CAPABILITIES_POINTER: u32 = 0x34;
/// MSI-X capability id.
const PCI_CAP_ID_MSIX: u8 = 0x11;

/// Host page size used to align `mmap`'d BAR sub-regions.
fn host_page_size() -> u64 {
    // SAFETY: `sysconf` with `_SC_PAGESIZE` is always safe to call and returns a positive value on
    // Linux.
    let ret = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    // Page size is always a positive power of two on Linux.
    u64::try_from(ret).unwrap_or(4096)
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    align_down(value + alignment - 1, alignment)
}

/// The kind of address space a memory BAR lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PciBarType {
    /// 32-bit memory BAR.
    Memory32,
    /// 64-bit memory BAR (consumes two BAR slots).
    Memory64,
}

/// A host `mmap` of a (sub-)range of a VFIO region, registered into the guest as a KVM memory
/// region so the guest accesses the device memory directly.
struct BarMmap {
    /// Host virtual address returned by `mmap`.
    host_addr: *mut libc::c_void,
    /// Length of the mapping in bytes.
    length: usize,
    /// Guest physical address this mapping is exposed at.
    guest_addr: u64,
    /// KVM memory slot used for this mapping.
    slot: u32,
}

// SAFETY: `BarMmap` owns a raw `mmap` pointer to shared device memory. The pointer is only ever
// handed to the kernel (`mmap`/`munmap`/`KVM_SET_USER_MEMORY_REGION`); it is never dereferenced
// from Rust and the device accesses it concurrently via DMA/MMIO, so it is safe to move the
// owning handle across threads.
unsafe impl Send for BarMmap {}

impl std::fmt::Debug for BarMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarMmap")
            .field("guest_addr", &format_args!("{:#x}", self.guest_addr))
            .field("length", &self.length)
            .field("slot", &self.slot)
            .finish()
    }
}

/// A single BAR of the assigned device as exposed to the guest.
#[derive(Debug)]
struct MmioRegion {
    /// VFIO region index, equal to the BAR index (0..=5).
    index: u32,
    /// Guest physical base address assigned to this BAR.
    guest_addr: u64,
    /// Size of the BAR in bytes.
    size: u64,
    /// Address space type of the BAR.
    bar_type: PciBarType,
    /// Direct mappings of the device memory into the guest (empty for fully trapped BARs).
    mmaps: Vec<BarMmap>,
}

/// Description of where the MSI-X table and PBA live, parsed from the device's MSI-X capability.
#[derive(Debug, Clone, Copy)]
struct MsixLayout {
    /// Byte offset of the MSI-X capability in configuration space.
    cap_offset: u16,
    /// BAR index holding the MSI-X table.
    table_bar: u32,
    /// Offset of the MSI-X table within its BAR.
    table_offset: u64,
    /// Size of the MSI-X table in bytes.
    table_size: u64,
    /// BAR index holding the PBA.
    pba_bar: u32,
    /// Offset of the PBA within its BAR.
    pba_offset: u64,
    /// Size of the PBA in bytes.
    pba_size: u64,
}

/// Errors that can occur while building or driving a [`VfioPciDevice`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VfioPciError {
    /// The device does not expose an MSI-X capability, which is required for interrupt delivery
    MissingMsix,
    /// Failed to allocate guest address space for a BAR: {0}
    BarAllocation(#[from] vm_allocator::Error),
    /// Failed to mmap a BAR region: {0}
    Mmap(#[source] std::io::Error),
    /// Ran out of KVM memory slots while mapping device BARs
    NotEnoughKvmSlots,
    /// Failed to register a BAR as a KVM memory region: {0}
    RegisterMemoryRegion(String),
    /// Failed to program the device's MSI-X interrupts: {0}
    Interrupt(#[source] vfio_ioctls::VfioError),
}

/// A physical PCI device assigned to the guest through VFIO.
pub struct VfioPciDevice {
    /// Firecracker id of the device.
    id: String,
    /// Segment/Bus/Device/Function assigned on the guest PCI bus.
    sbdf: PciSBDF,
    /// The VFIO container and device backing the passthrough.
    resources: VfioPciResources,
    /// The emulated BARs as seen by the guest. Used to answer configuration space BAR reads.
    bars: Bars,
    /// The enumerated MMIO regions (BARs) of the device.
    regions: Vec<MmioRegion>,
    /// Location of the MSI-X table and PBA.
    msix_layout: MsixLayout,
    /// The MSI-X vectors allocated for the device (irqfds + GSIs).
    msix_vectors: Arc<MsixVectorGroup>,
    /// Virtualized MSI-X configuration (table + PBA) shared with the interrupt path.
    msix_config: Arc<Mutex<MsixConfig>>,
    /// Whether MSI-X has been enabled on the physical device.
    msix_enabled: bool,
}

impl std::fmt::Debug for VfioPciDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VfioPciDevice")
            .field("id", &self.id)
            .field("sbdf", &self.sbdf)
            .field("regions", &self.regions)
            .field("msix_layout", &self.msix_layout)
            .field("msix_enabled", &self.msix_enabled)
            .finish_non_exhaustive()
    }
}

impl VfioPciDevice {
    /// Create a new passthrough device for the function described by `resources`.
    ///
    /// This parses the device's MSI-X capability, allocates the matching MSI-X vectors and sets up
    /// the virtualized MSI-X configuration. BAR allocation and mapping is performed separately in
    /// [`VfioPciDevice::allocate_and_map_bars`] once the device is attached to a segment.
    pub fn new(
        id: String,
        sbdf: PciSBDF,
        resources: VfioPciResources,
        msix_vectors: Arc<MsixVectorGroup>,
    ) -> Result<Self, VfioPciError> {
        let msix_layout =
            Self::parse_msix_capability(&resources.device).ok_or(VfioPciError::MissingMsix)?;

        let msix_config = Arc::new(Mutex::new(MsixConfig::new(Arc::clone(&msix_vectors), sbdf)));

        Ok(Self {
            id,
            sbdf,
            resources,
            bars: Bars::default(),
            regions: Vec::new(),
            msix_layout,
            msix_vectors,
            msix_config,
            msix_enabled: false,
        })
    }

    /// The Firecracker id of this device.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The host sysfs path of the assigned device.
    pub fn sysfs_path(&self) -> &std::path::Path {
        &self.resources.path
    }

    /// Number of MSI-X vectors the device requires, as reported by its MSI-X capability.
    ///
    /// Returns `None` if the device has no MSI-X capability.
    pub fn required_msix_vectors(device: &VfioDevice) -> Option<u16> {
        Self::parse_msix_capability(device)
            .map(|layout| u16::try_from(layout.table_size / MSIX_TABLE_ENTRY_SIZE).unwrap_or(0))
    }

    /// Read a dword from the device's configuration space.
    fn read_config_dword(device: &VfioDevice, offset: u32) -> u32 {
        let mut data = [0u8; 4];
        device.region_read(VFIO_PCI_CONFIG_REGION_INDEX, &mut data, u64::from(offset));
        u32::from_le_bytes(data)
    }

    /// Walk the device's PCI capability list and parse its MSI-X capability, if present.
    fn parse_msix_capability(device: &VfioDevice) -> Option<MsixLayout> {
        let status = Self::read_config_dword(device, 0x04);
        if status & PCI_CONFIG_STATUS_CAPABILITIES_LIST == 0 {
            return None;
        }

        // The capabilities pointer is the low byte at offset 0x34.
        let mut cap_next =
            u16::try_from(Self::read_config_dword(device, PCI_CONFIG_CAPABILITIES_POINTER) & 0xff)
                .unwrap();

        // Bound the walk to the 256 byte standard configuration space to avoid cycles.
        while cap_next != 0 && cap_next < 0x100 {
            let header = Self::read_config_dword(device, u32::from(cap_next));
            let cap_id = u8::try_from(header & 0xff).unwrap();
            let next = u16::try_from((header >> 8) & 0xff).unwrap();

            if cap_id == PCI_CAP_ID_MSIX {
                let msg_ctl = u16::try_from((header >> 16) & 0xffff).unwrap();
                let table = Self::read_config_dword(device, u32::from(cap_next) + 4);
                let pba = Self::read_config_dword(device, u32::from(cap_next) + 8);

                let num_vectors = u64::from((msg_ctl & 0x7ff) + 1);
                let table_size = num_vectors * MSIX_TABLE_ENTRY_SIZE;
                // The PBA has one bit per vector, rounded up to a multiple of 8 bytes.
                let pba_size = num_vectors.div_ceil(8);

                return Some(MsixLayout {
                    cap_offset: cap_next,
                    table_bar: table & 0x7,
                    table_offset: u64::from(table & 0xffff_fff8),
                    table_size,
                    pba_bar: pba & 0x7,
                    pba_offset: u64::from(pba & 0xffff_fff8),
                    pba_size,
                });
            }

            cap_next = next;
        }

        None
    }

    /// Allocate guest address space for every BAR and map the device memory into the guest.
    ///
    /// Memory BARs that the host kernel allows to be `mmap`'d are mapped directly as KVM memory
    /// regions for exit-less access; the MSI-X table/PBA pages and any non-mappable BARs are left
    /// to be trapped on the MMIO bus.
    pub fn allocate_and_map_bars(&mut self, vm: &Arc<KvmVm>) -> Result<(), VfioPciError> {
        let mut bar_index = 0u32;
        while bar_index < u32::from(NUM_BAR_REGS) {
            let region_index = VFIO_PCI_BAR0_REGION_INDEX + bar_index;
            let size = self.resources.device.get_region_size(region_index);
            if size == 0 {
                bar_index += 1;
                continue;
            }

            // Read the low bits of the real BAR to learn its type (memory vs IO, 32 vs 64 bit,
            // prefetchable). We only support memory BARs for passthrough; IO BARs are legacy and
            // not used by modern devices such as GPUs.
            let bar_cfg = Self::read_config_dword(
                &self.resources.device,
                PCI_CONFIG_BAR_OFFSET + bar_index * 4,
            );
            let is_io = (bar_cfg & 0x1) == 0x1;
            if is_io {
                warn!(
                    "vfio: {} BAR{bar_index} is an IO BAR which is not supported; skipping",
                    self.sbdf
                );
                bar_index += 1;
                continue;
            }
            let is_64bit = (bar_cfg & 0x6) == 0x4;
            let prefetchable = if (bar_cfg & 0x8) == 0x8 {
                BarPrefetchable::Yes
            } else {
                BarPrefetchable::No
            };
            let bar_type = if is_64bit {
                PciBarType::Memory64
            } else {
                PciBarType::Memory32
            };

            // Allocate guest address space for the BAR. 64-bit BARs go in the 64-bit MMIO window;
            // 32-bit BARs in the 32-bit window.
            let guest_addr = {
                let mut allocator = vm.resource_allocator();
                let window = if is_64bit {
                    &mut allocator.mmio64_memory
                } else {
                    &mut allocator.mmio32_memory
                };
                window
                    .allocate(size, size, vm_allocator::AllocPolicy::FirstMatch)?
                    .start()
            };

            let bar_idx = u8::try_from(bar_index).unwrap();
            match bar_type {
                PciBarType::Memory64 => {
                    self.bars
                        .set_bar_64(bar_idx, guest_addr, size, prefetchable)
                }
                PciBarType::Memory32 => {
                    self.bars
                        .set_bar_32(bar_idx, guest_addr, size, prefetchable)
                }
            }

            let mut region = MmioRegion {
                index: region_index,
                guest_addr,
                size,
                bar_type,
                mmaps: Vec::new(),
            };

            self.map_region(vm, &mut region)?;
            self.regions.push(region);

            // 64-bit BARs consume the next BAR slot for their high dword.
            bar_index += if is_64bit { 2 } else { 1 };
        }

        Ok(())
    }

    /// Map the `mmap`'able parts of a single BAR into the guest, carving out the MSI-X table and
    /// PBA pages so they can be trapped and virtualized.
    fn map_region(&self, vm: &Arc<KvmVm>, region: &mut MmioRegion) -> Result<(), VfioPciError> {
        let flags = self.resources.device.get_region_flags(region.index);
        if flags & VFIO_REGION_INFO_FLAG_MMAP == 0 {
            // The whole BAR must be trapped and forwarded to the device.
            return Ok(());
        }

        // Determine the base set of mappable areas. If the kernel reports a sparse mmap
        // capability, only those areas are mappable; otherwise the whole BAR is.
        let caps = self.resources.device.get_region_caps(region.index);
        let base_areas: Vec<(u64, u64)> = caps
            .iter()
            .find_map(|cap| match cap {
                VfioRegionInfoCap::SparseMmap(sparse) => {
                    Some(sparse.areas.iter().map(|a| (a.offset, a.size)).collect())
                }
                _ => None,
            })
            .unwrap_or_else(|| vec![(0, region.size)]);

        // Compute the page-aligned holes that must stay trapped (MSI-X table and PBA when they
        // live in this BAR).
        let page_size = host_page_size();
        let mut holes: Vec<(u64, u64)> = Vec::new();
        if self.msix_layout.table_bar == region.index {
            holes.push((
                align_down(self.msix_layout.table_offset, page_size),
                align_up(
                    self.msix_layout.table_offset + self.msix_layout.table_size,
                    page_size,
                ),
            ));
        }
        if self.msix_layout.pba_bar == region.index {
            holes.push((
                align_down(self.msix_layout.pba_offset, page_size),
                align_up(
                    self.msix_layout.pba_offset + self.msix_layout.pba_size,
                    page_size,
                ),
            ));
        }

        let region_offset = self.resources.device.get_region_offset(region.index);
        let mut prot = 0;
        if flags & VFIO_REGION_INFO_FLAG_READ != 0 {
            prot |= libc::PROT_READ;
        }
        if flags & VFIO_REGION_INFO_FLAG_WRITE != 0 {
            prot |= libc::PROT_WRITE;
        }

        for (base_offset, base_size) in base_areas {
            for (area_offset, area_size) in
                Self::subtract_holes(base_offset, base_offset + base_size, &holes)
            {
                let mmap = Self::mmap_area(
                    vm,
                    self.resources.device.as_raw_fd(),
                    region_offset + area_offset,
                    area_size,
                    prot,
                    region.guest_addr + area_offset,
                )?;
                region.mmaps.push(mmap);
            }
        }

        Ok(())
    }

    /// `mmap` a sub-area of a VFIO region and register it as a KVM memory region.
    fn mmap_area(
        vm: &Arc<KvmVm>,
        device_fd: RawFd,
        file_offset: u64,
        length: u64,
        prot: libc::c_int,
        guest_addr: u64,
    ) -> Result<BarMmap, VfioPciError> {
        let length_usize = usize::try_from(length).map_err(|_| {
            VfioPciError::Mmap(std::io::Error::other("BAR mapping length overflows usize"))
        })?;
        let file_offset = i64::try_from(file_offset).map_err(|_| {
            VfioPciError::Mmap(std::io::Error::other("BAR mapping offset overflows off_t"))
        })?;

        // SAFETY: We pass a null hint so the kernel chooses the address, a valid VFIO device fd and
        // an offset/length that VFIO reported as mmap'able for this region. The returned mapping is
        // owned by the `BarMmap` and unmapped on drop.
        let host_addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length_usize,
                prot,
                libc::MAP_SHARED,
                device_fd,
                file_offset,
            )
        };
        if host_addr == libc::MAP_FAILED {
            return Err(VfioPciError::Mmap(std::io::Error::last_os_error()));
        }

        let slot = vm.next_kvm_slot(1).ok_or(VfioPciError::NotEnoughKvmSlots)?;

        let kvm_region = kvm_bindings::kvm_userspace_memory_region {
            slot,
            flags: 0,
            guest_phys_addr: guest_addr,
            memory_size: length,
            userspace_addr: host_addr as u64,
        };

        if let Err(err) = vm.set_user_memory_region(kvm_region) {
            // SAFETY: `host_addr`/`length_usize` are the values returned by the `mmap` above.
            unsafe {
                libc::munmap(host_addr, length_usize);
            }
            return Err(VfioPciError::RegisterMemoryRegion(err.to_string()));
        }

        debug!(
            "vfio: mapped BAR area gpa={guest_addr:#x} len={length:#x} host={host_addr:?} \
             slot={slot}"
        );

        Ok(BarMmap {
            host_addr,
            length: length_usize,
            guest_addr,
            slot,
        })
    }

    /// Split the range `[start, end)` into the sub-ranges not covered by any of `holes`.
    ///
    /// Returns `(offset, size)` pairs.
    fn subtract_holes(start: u64, end: u64, holes: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut pieces = vec![(start, end)];
        for &(hole_start, hole_end) in holes {
            let mut next = Vec::new();
            for (piece_start, piece_end) in pieces {
                if hole_end <= piece_start || hole_start >= piece_end {
                    // No overlap.
                    next.push((piece_start, piece_end));
                } else {
                    if piece_start < hole_start {
                        next.push((piece_start, hole_start));
                    }
                    if hole_end < piece_end {
                        next.push((hole_end, piece_end));
                    }
                }
            }
            pieces = next;
        }
        pieces
            .into_iter()
            .filter(|(s, e)| e > s)
            .map(|(s, e)| (s, e - s))
            .collect()
    }

    /// Given a region of `size` bytes and a sorted, non-overlapping list of mapped ranges, return
    /// the complement: the ranges `(offset, len)` not covered by any mapping.
    fn complement_ranges(size: u64, mapped: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut areas = Vec::new();
        let mut cursor = 0u64;
        for &(start, len) in mapped {
            let start = start.min(size);
            let end = (start + len).min(size);
            if start > cursor {
                areas.push((cursor, start - cursor));
            }
            cursor = cursor.max(end);
        }
        if cursor < size {
            areas.push((cursor, size - cursor));
        }
        areas
    }

    /// The guest physical base addresses and sizes of the BAR ranges that must be trapped on the
    /// MMIO bus (the complement of the directly mapped ranges within each BAR).
    pub fn trapped_bus_ranges(&self) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        for region in &self.regions {
            let mut mapped: Vec<(u64, u64)> = region
                .mmaps
                .iter()
                .map(|m| (m.guest_addr - region.guest_addr, m.length as u64))
                .collect();
            mapped.sort_unstable();
            for (offset, len) in Self::complement_ranges(region.size, &mapped) {
                ranges.push((region.guest_addr + offset, len));
            }
        }
        ranges
    }

    /// Find the region (BAR) that contains the given guest physical address.
    fn region_for_addr(&self, addr: u64) -> Option<&MmioRegion> {
        self.regions
            .iter()
            .find(|r| addr >= r.guest_addr && addr < r.guest_addr + r.size)
    }

    /// Whether the given BAR offset falls within the MSI-X table.
    fn is_msix_table(&self, region_index: u32, offset: u64) -> bool {
        region_index == self.msix_layout.table_bar
            && offset >= self.msix_layout.table_offset
            && offset < self.msix_layout.table_offset + self.msix_layout.table_size
    }

    /// Whether the given BAR offset falls within the MSI-X PBA.
    fn is_msix_pba(&self, region_index: u32, offset: u64) -> bool {
        region_index == self.msix_layout.pba_bar
            && offset >= self.msix_layout.pba_offset
            && offset < self.msix_layout.pba_offset + self.msix_layout.pba_size
    }

    /// Program the physical device's MSI-X interrupts so they are delivered to the guest through
    /// the MSI-X vectors' irqfds.
    fn enable_msix(&mut self) -> Result<(), VfioPciError> {
        let event_fds: Vec<&EventFd> = (0..self.msix_vectors.vectors.len())
            .filter_map(|i| self.msix_vectors.notifier(i))
            .collect();

        self.resources
            .device
            .enable_irq(VFIO_PCI_MSIX_IRQ_INDEX, event_fds)
            .map_err(VfioPciError::Interrupt)?;
        self.msix_enabled = true;
        debug!(
            "vfio: {} enabled MSI-X with {} vectors",
            self.sbdf,
            self.msix_vectors.vectors.len()
        );
        Ok(())
    }

    /// Disable the physical device's MSI-X interrupts.
    fn disable_msix(&mut self) {
        if let Err(err) = self.resources.device.disable_irq(VFIO_PCI_MSIX_IRQ_INDEX) {
            error!("vfio: {} failed to disable MSI-X: {err}", self.sbdf);
        }
        self.msix_enabled = false;
    }

    /// React to a guest write to the MSI-X capability's message control register: enable or
    /// disable MSI-X on the physical device as the guest toggles the enable bit.
    fn update_msix_capability(&mut self) {
        let enabled = self.msix_config.lock().expect("Poisoned lock").enabled;
        if enabled && !self.msix_enabled {
            if let Err(err) = self.enable_msix() {
                error!("vfio: {} failed to enable MSI-X: {err}", self.sbdf);
            }
        } else if !enabled && self.msix_enabled {
            self.disable_msix();
        }
    }
}

impl PciDevice for VfioPciDevice {
    fn write_config_register(
        &mut self,
        reg_idx: u16,
        offset: u8,
        data: &[u8],
    ) -> Option<Arc<Barrier>> {
        let in_bars = (PCI_CONFIG_BAR0_INDEX..PCI_CONFIG_BAR0_INDEX + u16::from(NUM_BAR_REGS))
            .contains(&reg_idx);
        if in_bars {
            // Trap BAR writes into the virtualized BAR registers; never reprogram the real device.
            let bar_idx = u8::try_from(reg_idx - PCI_CONFIG_BAR0_INDEX).unwrap();
            self.bars.write(bar_idx, offset, data);
            return None;
        }
        if reg_idx == PCI_CONFIG_ROM_BAR_INDEX {
            // The expansion ROM BAR is not exposed to the guest.
            return None;
        }

        // Detect writes to the MSI-X message control register (capability offset + 2). The control
        // register carries the MSI-X enable and function mask bits.
        let msg_ctl_offset = u32::from(self.msix_layout.cap_offset) + 2;
        let write_start = u32::from(reg_idx) * 4 + u32::from(offset);
        let write_end = write_start + u32::try_from(data.len()).unwrap();
        let touches_msg_ctl = write_start < msg_ctl_offset + 2 && write_end > msg_ctl_offset;

        // Pass the write through to the physical device's configuration space.
        self.resources.device.region_write(
            VFIO_PCI_CONFIG_REGION_INDEX,
            data,
            u64::from(write_start),
        );

        if touches_msg_ctl {
            // Mirror the control register into the virtualized MSI-X config and (un)route the
            // physical interrupts accordingly.
            let ctl_in_write = usize::try_from(msg_ctl_offset.saturating_sub(write_start)).unwrap();
            if ctl_in_write + 2 <= data.len() {
                let msg_ctl = u16::from_le_bytes([data[ctl_in_write], data[ctl_in_write + 1]]);
                self.msix_config
                    .lock()
                    .expect("Poisoned lock")
                    .set_msg_ctl(msg_ctl);
            }
            self.update_msix_capability();
        }

        None
    }

    fn read_config_register(&mut self, reg_idx: u16) -> u32 {
        let in_bars = (PCI_CONFIG_BAR0_INDEX..PCI_CONFIG_BAR0_INDEX + u16::from(NUM_BAR_REGS))
            .contains(&reg_idx);
        if in_bars {
            let bar_idx = u8::try_from(reg_idx - PCI_CONFIG_BAR0_INDEX).unwrap();
            let mut value = [0u8; 4];
            self.bars.read(bar_idx, 0, &mut value);
            return u32::from_le_bytes(value);
        }
        if reg_idx == PCI_CONFIG_ROM_BAR_INDEX {
            return 0;
        }
        // Everything else is read straight from the physical device.
        Self::read_config_dword(&self.resources.device, u32::from(reg_idx) * 4)
    }

    fn read_bar(&mut self, base: u64, offset: u64, data: &mut [u8]) {
        let Some(region) = self.region_for_addr(base) else {
            warn!("vfio: {} read to unmapped BAR base {base:#x}", self.sbdf);
            data.fill(0);
            return;
        };
        let region_index = region.index;
        let bar_offset = base - region.guest_addr + offset;

        if self.is_msix_table(region_index, bar_offset) {
            self.msix_config
                .lock()
                .expect("Poisoned lock")
                .read_table(bar_offset - self.msix_layout.table_offset, data);
        } else if self.is_msix_pba(region_index, bar_offset) {
            self.msix_config
                .lock()
                .expect("Poisoned lock")
                .read_pba(bar_offset - self.msix_layout.pba_offset, data);
        } else {
            self.resources
                .device
                .region_read(region_index, data, bar_offset);
        }
    }

    fn write_bar(&mut self, base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        let Some(region) = self.region_for_addr(base) else {
            warn!("vfio: {} write to unmapped BAR base {base:#x}", self.sbdf);
            return None;
        };
        let region_index = region.index;
        let bar_offset = base - region.guest_addr + offset;

        if self.is_msix_table(region_index, bar_offset) {
            self.msix_config
                .lock()
                .expect("Poisoned lock")
                .write_table(bar_offset - self.msix_layout.table_offset, data);
        } else if self.is_msix_pba(region_index, bar_offset) {
            self.msix_config
                .lock()
                .expect("Poisoned lock")
                .write_pba(bar_offset - self.msix_layout.pba_offset, data);
        } else {
            self.resources
                .device
                .region_write(region_index, data, bar_offset);
        }
        None
    }
}

impl BusDevice for VfioPciDevice {
    fn read(&mut self, base: u64, offset: u64, data: &mut [u8]) {
        self.read_bar(base, offset, data)
    }

    fn write(&mut self, base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        self.write_bar(base, offset, data)
    }
}

impl Drop for BarMmap {
    fn drop(&mut self) {
        // SAFETY: `host_addr` and `length` are exactly the values returned by `mmap` in
        // `VfioPciDevice::mmap_area`; the mapping is owned exclusively by this `BarMmap`.
        unsafe {
            libc::munmap(self.host_addr, self.length);
        }
    }
}

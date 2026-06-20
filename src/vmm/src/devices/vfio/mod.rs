// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low level building blocks for VFIO based PCIe device passthrough.
//!
//! The Linux VFIO framework (<https://docs.kernel.org/driver-api/vfio.html>) exposes a physical
//! PCI device to userspace in an IOMMU protected fashion, which is exactly what is needed to
//! assign a host device (for example a GPU) to a Firecracker microVM. This module wraps the
//! [`vfio_ioctls`] crate and ties it into Firecracker's [`KvmVm`], providing:
//!
//! * creation of a VFIO container (an IOMMU domain) coupled to KVM through the KVM-VFIO pseudo
//!   device, so that KVM is aware of the assigned device (required for correct handling of
//!   non-coherent DMA and interrupt remapping),
//! * opening the assigned VFIO device given its sysfs path,
//! * mapping the whole guest physical address space into the IOMMU so the device can DMA into
//!   guest RAM.
//!
//! The higher level PCI emulation (configuration space, BARs and MSI-X) that turns a
//! [`VfioDevice`] into a device the guest can drive lives in [`pci`].

pub mod pci;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kvm_bindings::{kvm_create_device, kvm_device_type_KVM_DEV_TYPE_VFIO};
use vfio_ioctls::{VfioContainer, VfioDevice, VfioDeviceFd};

use crate::vstate::vm::KvmVm;

/// Errors that can occur while setting up a VFIO passthrough device.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VfioError {
    /// Failed to create the KVM VFIO pseudo device: {0}
    CreateKvmDevice(#[source] kvm_ioctls::Error),
    /// Failed to create the VFIO container: {0}
    CreateContainer(#[source] vfio_ioctls::VfioError),
    /// Failed to open the VFIO device at `{1}`: {0}
    OpenDevice(#[source] vfio_ioctls::VfioError, PathBuf),
    /// Failed to map guest memory into the IOMMU for DMA: {0}
    DmaMap(#[source] vfio_ioctls::VfioError),
    /// Failed to unmap guest memory from the IOMMU: {0}
    DmaUnmap(#[source] vfio_ioctls::VfioError),
}

/// A physical PCI device assigned to the guest through VFIO, together with the IOMMU container
/// backing its DMA mappings.
///
/// The container owns the IOMMU domain and the guest memory DMA mappings, while [`VfioDevice`]
/// exposes the device's regions (config space, BARs) and interrupts. They are kept together
/// because the device's lifetime is bound to the container: dropping the container tears down the
/// IOMMU mappings that make the device safe to use.
pub struct VfioPciResources {
    /// The IOMMU container (a single IOMMU domain). It is reference counted because
    /// [`VfioDevice`] keeps a clone of it internally as its [`vfio_ioctls::VfioOps`] backend.
    pub container: Arc<VfioContainer>,
    /// The underlying VFIO device handle used for region and interrupt access.
    pub device: Arc<VfioDevice>,
    /// The host sysfs path of the assigned device, e.g.
    /// `/sys/bus/pci/devices/0000:01:00.0`.
    pub path: PathBuf,
}

impl std::fmt::Debug for VfioPciResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `VfioContainer` and `VfioDevice` do not implement `Debug` and wrap raw file
        // descriptors, so only the stable, meaningful identity (the sysfs path) is shown.
        f.debug_struct("VfioPciResources")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl VfioPciResources {
    /// Assign the host PCI device located at `sysfs_path` to the guest owned by `vm`.
    ///
    /// This creates a fresh IOMMU container coupled to KVM, opens the device (binding its IOMMU
    /// group to the container) and maps the entire current guest physical memory into the IOMMU
    /// so the device can perform DMA to and from guest RAM.
    ///
    /// `sysfs_path` must point at the device's sysfs directory (the one containing the
    /// `iommu_group` symlink), and the device must already be bound to the `vfio-pci` driver on
    /// the host.
    pub fn new(vm: &KvmVm, sysfs_path: &Path) -> Result<Self, VfioError> {
        let container = create_kvm_coupled_container(vm)?;

        // `VfioDevice` takes its IOMMU backend as an `Arc<dyn VfioOps>`; coerce the concrete
        // container into the trait object (the container keeps its own clone internally).
        let vfio_ops: Arc<dyn vfio_ioctls::VfioOps> = Arc::clone(&container) as _;
        let device = VfioDevice::new(sysfs_path, vfio_ops)
            .map_err(|err| VfioError::OpenDevice(err, sysfs_path.to_path_buf()))?;

        // Map the whole guest physical address space into the IOMMU. After this, the assigned
        // device can DMA into any guest RAM page, exactly like it could access host RAM when
        // running on bare metal. The IOMMU guarantees the device cannot reach memory outside of
        // these mappings.
        //
        // SAFETY: `guest_memory` is backed by host mmap regions that stay pinned and at a fixed
        // host virtual address for the entire lifetime of the VM (Firecracker never moves or
        // unmaps guest RAM while the VM is running). We never create Rust references aliasing this
        // memory while the device has DMA access to it, and the mappings are torn down when the
        // container is dropped.
        unsafe {
            container
                .vfio_map_guest_memory(vm.guest_memory())
                .map_err(VfioError::DmaMap)?;
        }

        Ok(Self {
            container,
            device: Arc::new(device),
            path: sysfs_path.to_path_buf(),
        })
    }
}

/// Create a VFIO container (IOMMU domain) and couple it to KVM through the KVM-VFIO pseudo
/// device.
///
/// Coupling the container to KVM (via `KVM_CREATE_DEVICE` of type `KVM_DEV_TYPE_VFIO` and, inside
/// [`VfioContainer`], `KVM_DEV_VFIO_GROUP_ADD`) lets KVM track the assigned device. This is
/// required for KVM to correctly handle device originated DMA that bypasses the cache (so that
/// `WBINVD` is emulated when a non-coherent device is assigned) and to set up interrupt
/// remapping.
fn create_kvm_coupled_container(vm: &KvmVm) -> Result<Arc<VfioContainer>, VfioError> {
    let mut vfio_device = kvm_create_device {
        type_: kvm_device_type_KVM_DEV_TYPE_VFIO,
        fd: 0,
        flags: 0,
    };
    let device_fd = vm
        .fd()
        .create_device(&mut vfio_device)
        .map_err(VfioError::CreateKvmDevice)?;

    // The container keeps the device fd alive for as long as it lives; the KVM-VFIO coupling is
    // dropped together with the container.
    let device_fd = Arc::new(VfioDeviceFd::new_from_kvm(device_fd));

    let container = VfioContainer::new(Some(device_fd)).map_err(VfioError::CreateContainer)?;

    Ok(Arc::new(container))
}

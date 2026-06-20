# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VFIO based PCIe device passthrough.

The full passthrough path needs a real host PCI device bound to the ``vfio-pci``
driver, which is not available on the CI fleet. Those tests are therefore gated
behind the ``FC_TEST_VFIO_DEVICE`` environment variable (the host sysfs path of
an assigned device, e.g. ``/sys/bus/pci/devices/0000:01:00.0``) and the presence
of ``/dev/vfio/vfio``; they skip cleanly otherwise. The input-validation test
below does not need any passthrough hardware and always runs.
"""

import os
from pathlib import Path

import pytest

from framework.artifacts import pin_pci

VFIO_DEVICE = os.environ.get("FC_TEST_VFIO_DEVICE")


def _vfio_available():
    """Whether a real VFIO passthrough device is available to test against."""
    return (
        VFIO_DEVICE is not None
        and Path(VFIO_DEVICE).exists()
        and Path("/dev/vfio/vfio").exists()
    )


needs_vfio = pytest.mark.skipif(
    not _vfio_available(),
    reason=(
        "no VFIO device available; set FC_TEST_VFIO_DEVICE to the sysfs path of "
        "a host PCI device bound to vfio-pci"
    ),
)


@pin_pci(True)
def test_vfio_request_validation(uvm):
    """The /vfio endpoint rejects an invalid request body (no hardware needed)."""
    vm = uvm
    vm.spawn()
    vm.basic_config()

    # The `path` field is required, so a request without it must be rejected.
    with pytest.raises(RuntimeError):
        vm.api.vfio.put(id="dev0")


@needs_vfio
@pin_pci(True)
def test_vfio_passthrough(uvm):
    """Assign a real host PCI device to the guest and check it is visible."""
    vm = uvm
    vm.spawn()
    vm.basic_config()
    vm.add_net_iface()

    vm.api.vfio.put(id="passthrough0", path=VFIO_DEVICE)
    vm.start()

    # The assigned device must show up on the guest PCI bus, in addition to the
    # PCI host bridge.
    lspci = vm.ssh.run("lspci -D").stdout.strip()
    assert lspci, "no PCI devices found in guest"
    assert len(lspci.splitlines()) >= 2, lspci


@needs_vfio
@pin_pci(True)
def test_vfio_blocks_snapshot(uvm):
    """A microVM with a passthrough device cannot be snapshotted."""
    vm = uvm
    vm.spawn()
    vm.basic_config()
    vm.api.vfio.put(id="passthrough0", path=VFIO_DEVICE)
    vm.start()

    vm.api.vm.patch(state="Paused")
    with pytest.raises(RuntimeError):
        vm.api.snapshot_create.put(
            mem_file_path="/mem.snap",
            snapshot_path="/state.snap",
        )

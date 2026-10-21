package topology_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/topology"
)

// eightGPUHost is a plausible node: two NVLink islands of four devices each,
// which is the layout that makes interconnect-aware placement matter.
func eightGPUHost(t *testing.T, host string, memMB int) *topology.Fleet {
	t.Helper()
	f := topology.NewFleet()
	for i := 0; i < 8; i++ {
		domain := fmt.Sprintf("%s-nvl%d", host, i/4)
		if err := f.AddDevice(topology.Device{
			ID:            fmt.Sprintf("%s-gpu%d", host, i),
			Host:          host,
			Model:         "H100-80GB",
			MemoryMB:      memMB,
			Domain:        domain,
			Link:          topology.LinkNVLink,
			ComputeTFlops: 989,
		}); err != nil {
			t.Fatal(err)
		}
	}
	return f
}

func TestSingleDevicePlacement(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)

	p, err := f.Place(topology.Request{ID: "m1", Model: "llama-8b", MemoryMB: 20_000})
	if err != nil {
		t.Fatalf("Place: %v", err)
	}
	if len(p.Devices) != 1 {
		t.Fatalf("placed on %d devices, want 1", len(p.Devices))
	}

	var used int
	for _, d := range f.Status() {
		used += d.UsedMB
	}
	if used != 20_000 {
		t.Fatalf("fleet reports %d MB used, want 20000", used)
	}
}

// TestTensorParallelStaysInOneDomain is the placement rule that actually
// matters: a four-way replica split across two NVLink islands would work and be
// slow, which is the worst kind of bug.
func TestTensorParallelStaysInOneDomain(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)

	p, err := f.Place(topology.Request{
		ID: "tp4", Model: "llama-70b", DeviceCount: 4, MemoryMB: 40_000,
	})
	if err != nil {
		t.Fatalf("Place: %v", err)
	}
	if len(p.Devices) != 4 {
		t.Fatalf("placed on %d devices, want 4", len(p.Devices))
	}

	domains := map[string]struct{}{}
	byID := map[string]topology.DeviceStatus{}
	for _, d := range f.Status() {
		byID[d.ID] = d
	}
	for _, id := range p.Devices {
		domains[byID[id].Domain] = struct{}{}
	}
	if len(domains) != 1 {
		t.Fatalf("replica spans %d interconnect domains: %v", len(domains), domains)
	}
	if p.Link != topology.LinkNVLink {
		t.Fatalf("Link = %q, want nvlink", p.Link)
	}
}

func TestNoDomainLargeEnough(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)
	// Eight-way parallelism cannot fit in a four-device island.
	_, err := f.Place(topology.Request{ID: "tp8", DeviceCount: 8, MemoryMB: 10_000})
	if !errors.Is(err, topology.ErrNoDomain) {
		t.Fatalf("got %v, want ErrNoDomain", err)
	}
}

func TestNoCapacity(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)
	_, err := f.Place(topology.Request{ID: "huge", MemoryMB: 200_000})
	if !errors.Is(err, topology.ErrNoCapacity) {
		t.Fatalf("got %v, want ErrNoCapacity", err)
	}
}

// TestBestFitPacking checks the packing rule. Filling the emptiest device first
// leaves every domain half used and unable to take a large replica; best fit
// keeps a domain free.
func TestBestFitPacking(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)

	// Seed one device so the domains differ in free memory.
	if _, err := f.Place(topology.Request{ID: "seed", MemoryMB: 10_000}); err != nil {
		t.Fatal(err)
	}
	seeded := f.Placements()[0].Devices[0]

	// The next small replica should land on the already-used device, not on a
	// pristine one.
	p, err := f.Place(topology.Request{ID: "second", MemoryMB: 10_000})
	if err != nil {
		t.Fatal(err)
	}
	if p.Devices[0] != seeded {
		t.Fatalf("second replica went to %s, want the already-used %s (best fit)", p.Devices[0], seeded)
	}

	// A four-way replica needing most of each device must still fit, because
	// packing kept a whole island free.
	if _, err := f.Place(topology.Request{ID: "tp4", DeviceCount: 4, MemoryMB: 70_000}); err != nil {
		t.Fatalf("large replica did not fit after packing: %v", err)
	}
}

func TestPreferFasterInterconnect(t *testing.T) {
	f := topology.NewFleet()
	// A PCIe pair and an NVLink pair with identical free memory.
	for i := 0; i < 2; i++ {
		f.AddDevice(topology.Device{
			ID: fmt.Sprintf("pcie-%d", i), Host: "h", MemoryMB: 40_000,
			Domain: "pcie-dom", Link: topology.LinkPCIe,
		})
		f.AddDevice(topology.Device{
			ID: fmt.Sprintf("nvl-%d", i), Host: "h", MemoryMB: 40_000,
			Domain: "nvl-dom", Link: topology.LinkNVLink,
		})
	}
	p, err := f.Place(topology.Request{ID: "tp2", DeviceCount: 2, MemoryMB: 10_000})
	if err != nil {
		t.Fatal(err)
	}
	if p.Link != topology.LinkNVLink {
		t.Fatalf("chose %q over nvlink at equal capacity", p.Link)
	}
}

func TestRequireHost(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)
	for _, d := range eightGPUHost(t, "node-b", 80_000).Status() {
		f.AddDevice(d.Device)
	}

	p, err := f.Place(topology.Request{ID: "pinned", MemoryMB: 1000, RequireHost: "node-b"})
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]topology.DeviceStatus{}
	for _, d := range f.Status() {
		byID[d.ID] = d
	}
	if got := byID[p.Devices[0]].Host; got != "node-b" {
		t.Fatalf("placed on host %s, want node-b", got)
	}
}

func TestReleaseReturnsCapacity(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)

	if _, err := f.Place(topology.Request{ID: "big", DeviceCount: 4, MemoryMB: 79_000}); err != nil {
		t.Fatal(err)
	}
	before := f.Utilisation()
	if before <= 0 {
		t.Fatal("utilisation did not move after a placement")
	}

	if err := f.Release("big"); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if got := f.Utilisation(); got != 0 {
		t.Fatalf("utilisation = %v after releasing everything, want 0", got)
	}
	if len(f.Placements()) != 0 {
		t.Fatal("placement survived Release")
	}

	// And the freed room must be usable again.
	if _, err := f.Place(topology.Request{ID: "again", DeviceCount: 4, MemoryMB: 79_000}); err != nil {
		t.Fatalf("could not reuse released capacity: %v", err)
	}
}

func TestDuplicateAndUnknown(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)
	if _, err := f.Place(topology.Request{ID: "x", MemoryMB: 100}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Place(topology.Request{ID: "x", MemoryMB: 100}); err == nil {
		t.Fatal("duplicate placement ID was accepted")
	}
	if err := f.Release("nope"); !errors.Is(err, topology.ErrUnknownPlacement) {
		t.Fatalf("got %v, want ErrUnknownPlacement", err)
	}
}

// TestReInventoryPreservesUsage guards a real operational hazard: a periodic
// device discovery loop re-adding known devices must not zero their usage.
func TestReInventoryPreservesUsage(t *testing.T) {
	f := eightGPUHost(t, "node-a", 80_000)
	if _, err := f.Place(topology.Request{ID: "p", MemoryMB: 30_000}); err != nil {
		t.Fatal(err)
	}
	before := f.Utilisation()

	for _, d := range f.Status() {
		if err := f.AddDevice(d.Device); err != nil {
			t.Fatal(err)
		}
	}
	if got := f.Utilisation(); got != before {
		t.Fatalf("utilisation changed from %v to %v across re-inventory", before, got)
	}
}

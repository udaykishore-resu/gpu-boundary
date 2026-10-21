// Package topology models accelerator inventory and places model replicas onto
// it.
//
// Placement is where a control plane earns or loses most of its efficiency, and
// it is almost entirely a graph and constraint problem rather than a numerical
// one. Two facts drive it:
//
//   - Memory is the binding constraint. A replica either fits in the free HBM
//     on a device or it does not, and there is no graceful degradation.
//   - Interconnect is not uniform. Two GPUs on the same NVLink island exchange
//     tensors at a different order of magnitude than two GPUs that have to go
//     over PCIe or across hosts, so a tensor-parallel replica that spans
//     islands can be several times slower than one that does not.
//
// A placer that ignores the second fact will happily produce a technically
// valid, badly performing layout. This one does not: multi-device replicas are
// only placed on devices within one interconnect domain.
package topology

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Errors returned by the placer.
var (
	// ErrNoCapacity means nothing in the fleet can hold the replica.
	ErrNoCapacity = errors.New("topology: no device has capacity")
	// ErrNoDomain means no single interconnect domain has enough devices with
	// enough free memory to hold a multi-device replica.
	ErrNoDomain = errors.New("topology: no interconnect domain can hold the replica")
	// ErrUnknownDevice means the device ID is not registered.
	ErrUnknownDevice = errors.New("topology: unknown device")
	// ErrUnknownPlacement means the placement ID is not known.
	ErrUnknownPlacement = errors.New("topology: unknown placement")
)

// Link describes how a group of devices are connected to each other.
type Link string

// Interconnect classes, fastest first.
const (
	LinkNVLink Link = "nvlink"
	LinkPCIe   Link = "pcie"
	LinkNet    Link = "network"
)

// Bandwidth is an indicative GB/s figure used for ranking, not for capacity
// planning. The ratios are what matter here, not the absolute values.
func (l Link) Bandwidth() float64 {
	switch l {
	case LinkNVLink:
		return 900
	case LinkPCIe:
		return 64
	default:
		return 12.5
	}
}

// Device is one accelerator.
type Device struct {
	// ID is stable across restarts, typically host plus index.
	ID string
	// Host is the node the device is attached to.
	Host string
	// Model is the accelerator model, for reporting.
	Model string
	// MemoryMB is total device memory.
	MemoryMB int
	// Domain groups devices that share a fast interconnect. Devices in the
	// same domain are assumed mutually reachable at Link speed.
	Domain string
	// Link is the interconnect class within the domain.
	Link Link
	// ComputeTFlops ranks devices in a heterogeneous fleet.
	ComputeTFlops float64
}

// Placement is a model replica occupying one or more devices.
type Placement struct {
	// ID identifies the replica.
	ID string
	// Model is the served model.
	Model string
	// Devices are the device IDs the replica occupies. More than one means
	// tensor parallelism, and all of them are in the same domain.
	Devices []string
	// MemoryMB is the per-device memory reserved.
	MemoryMB int
	// Domain is the interconnect domain the placement landed in.
	Domain string
	// Link is that domain's interconnect class.
	Link Link
}

// Request asks for a placement.
type Request struct {
	// ID for the resulting placement.
	ID string
	// Model to serve.
	Model string
	// DeviceCount is the tensor parallel degree.
	DeviceCount int
	// MemoryMB required on each device.
	MemoryMB int
	// RequireHost, if set, constrains placement to one host.
	RequireHost string
}

type deviceState struct {
	Device
	usedMB int
	placed map[string]struct{}
}

func (d *deviceState) freeMB() int { return d.MemoryMB - d.usedMB }

// Fleet is the accelerator inventory.
type Fleet struct {
	mu         sync.RWMutex
	devices    map[string]*deviceState
	order      []string
	placements map[string]*Placement
}

// NewFleet returns an empty fleet.
func NewFleet() *Fleet {
	return &Fleet{
		devices:    make(map[string]*deviceState),
		placements: make(map[string]*Placement),
	}
}

// AddDevice registers a device. Re-adding a known device updates its static
// attributes and preserves its current usage, so a re-inventory does not
// silently free memory that is still in use.
func (f *Fleet) AddDevice(d Device) error {
	if d.ID == "" {
		return errors.New("topology: device ID must not be empty")
	}
	if d.MemoryMB <= 0 {
		return fmt.Errorf("topology: device %s has no memory", d.ID)
	}
	if d.Domain == "" {
		d.Domain = d.Host
	}
	if d.Link == "" {
		d.Link = LinkPCIe
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	if st, ok := f.devices[d.ID]; ok {
		st.Device = d
		return nil
	}
	f.devices[d.ID] = &deviceState{Device: d, placed: make(map[string]struct{})}
	f.order = append(f.order, d.ID)
	sort.Strings(f.order)
	return nil
}

// Place finds room for a replica and reserves it.
//
// The search is: group candidate devices by domain, discard domains that cannot
// supply enough devices with enough free memory, then prefer the fastest
// interconnect and, among equals, the domain with the least free memory. That
// last rule is best-fit rather than worst-fit, and it is deliberate: packing
// replicas tightly keeps whole domains empty for the next large model instead
// of leaving every domain half full and unable to take anything big.
func (f *Fleet) Place(req Request) (*Placement, error) {
	if req.DeviceCount <= 0 {
		req.DeviceCount = 1
	}
	if req.MemoryMB <= 0 {
		return nil, errors.New("topology: placement must request memory")
	}
	if req.ID == "" {
		return nil, errors.New("topology: placement ID must not be empty")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if _, exists := f.placements[req.ID]; exists {
		return nil, fmt.Errorf("topology: placement %s already exists", req.ID)
	}

	byDomain := make(map[string][]*deviceState)
	for _, id := range f.order {
		st := f.devices[id]
		if req.RequireHost != "" && st.Host != req.RequireHost {
			continue
		}
		if st.freeMB() < req.MemoryMB {
			continue
		}
		byDomain[st.Domain] = append(byDomain[st.Domain], st)
	}

	type candidate struct {
		domain  string
		devices []*deviceState
		link    Link
		freeMB  int
	}
	var cands []candidate
	for dom, devs := range byDomain {
		if len(devs) < req.DeviceCount {
			continue
		}
		// Within a domain, take the devices with the least free memory that
		// still fit, for the same best-fit reason.
		sort.Slice(devs, func(i, j int) bool {
			if devs[i].freeMB() != devs[j].freeMB() {
				return devs[i].freeMB() < devs[j].freeMB()
			}
			return devs[i].ID < devs[j].ID
		})
		chosen := devs[:req.DeviceCount]
		free := 0
		for _, d := range chosen {
			free += d.freeMB()
		}
		cands = append(cands, candidate{
			domain:  dom,
			devices: chosen,
			link:    chosen[0].Link,
			freeMB:  free,
		})
	}

	if len(cands) == 0 {
		if req.DeviceCount > 1 {
			return nil, fmt.Errorf("%w: %d devices with %d MB each",
				ErrNoDomain, req.DeviceCount, req.MemoryMB)
		}
		return nil, fmt.Errorf("%w: %d MB", ErrNoCapacity, req.MemoryMB)
	}

	sort.Slice(cands, func(i, j int) bool {
		bi, bj := cands[i].link.Bandwidth(), cands[j].link.Bandwidth()
		if bi != bj {
			return bi > bj
		}
		if cands[i].freeMB != cands[j].freeMB {
			return cands[i].freeMB < cands[j].freeMB
		}
		return cands[i].domain < cands[j].domain
	})

	win := cands[0]
	p := &Placement{
		ID:       req.ID,
		Model:    req.Model,
		MemoryMB: req.MemoryMB,
		Domain:   win.domain,
		Link:     win.link,
	}
	for _, d := range win.devices {
		d.usedMB += req.MemoryMB
		d.placed[req.ID] = struct{}{}
		p.Devices = append(p.Devices, d.ID)
	}
	sort.Strings(p.Devices)
	f.placements[req.ID] = p
	return p, nil
}

// Release frees a placement.
func (f *Fleet) Release(id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	p, ok := f.placements[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrUnknownPlacement, id)
	}
	for _, devID := range p.Devices {
		if st, ok := f.devices[devID]; ok {
			st.usedMB -= p.MemoryMB
			if st.usedMB < 0 {
				st.usedMB = 0
			}
			delete(st.placed, id)
		}
	}
	delete(f.placements, id)
	return nil
}

// DeviceStatus reports one device's utilisation.
type DeviceStatus struct {
	Device
	UsedMB     int
	FreeMB     int
	Placements []string
}

// Status reports the fleet, sorted by device ID.
func (f *Fleet) Status() []DeviceStatus {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]DeviceStatus, 0, len(f.order))
	for _, id := range f.order {
		st := f.devices[id]
		ps := make([]string, 0, len(st.placed))
		for p := range st.placed {
			ps = append(ps, p)
		}
		sort.Strings(ps)
		out = append(out, DeviceStatus{
			Device:     st.Device,
			UsedMB:     st.usedMB,
			FreeMB:     st.freeMB(),
			Placements: ps,
		})
	}
	return out
}

// Placements lists current placements, sorted by ID.
func (f *Fleet) Placements() []Placement {
	f.mu.RLock()
	defer f.mu.RUnlock()
	ids := make([]string, 0, len(f.placements))
	for id := range f.placements {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]Placement, 0, len(ids))
	for _, id := range ids {
		out = append(out, *f.placements[id])
	}
	return out
}

// Utilisation reports fleet-wide memory usage as a fraction.
func (f *Fleet) Utilisation() float64 {
	f.mu.RLock()
	defer f.mu.RUnlock()
	total, used := 0, 0
	for _, st := range f.devices {
		total += st.MemoryMB
		used += st.usedMB
	}
	if total == 0 {
		return 0
	}
	return float64(used) / float64(total)
}

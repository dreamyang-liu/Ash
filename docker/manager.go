package docker

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/multiturn-rl-hostagent/model"
	"github.com/multiturn-rl-hostagent/utils"
	"go.uber.org/zap"
)

// ContainerOptions represents options for creating a container
type ContainerOptions struct {
	Name                     string
	Env                      []string
	Ports                    map[string]string // host:container
	Command                  []string
	User                     string
	WorkingDir               string
	NetworkDisabled          bool
	BaseCommit               string
	EnvironmentSetupCommands []string
	Labels                   map[string]string
}

// ContainerStats represents resource usage statistics for a container
type ContainerStats struct {
	ContainerID string
	Name        string
	CPUUsage    float64 // percentage
	MemoryUsage float64 // percentage
	MemoryUsed  uint64  // bytes
	MemoryLimit uint64  // bytes
	DiskRead    uint64  // bytes
	DiskWrite   uint64  // bytes
}

// Manager handles Docker container operations
type Manager struct {
	client                *client.Client
	logger                *zap.Logger
	requestQueue          chan model.RolloutRequest
	responseQueue         chan model.RolloutResponse
	trajectoryInstanceMap map[string]string // Map of trajectory ID to container ID
}

// NewManager creates a new Docker manager
func NewManager(requestQueue chan model.RolloutRequest, responseQueue chan model.RolloutResponse) (*Manager, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create zap logger: %w", err)
	}

	return &Manager{
		client:                cli,
		logger:                logger,
		requestQueue:          requestQueue,
		responseQueue:         responseQueue,
		trajectoryInstanceMap: make(map[string]string),
	}, nil
}

func buildContainerOptions(request *model.RolloutRequest) ContainerOptions {
	return ContainerOptions{
		Name:                     request.TrajectoryID,
		Command:                  request.Commands,
		User:                     request.User,
		WorkingDir:               request.WorkingDir,
		NetworkDisabled:          request.NetworkDisabled,
		BaseCommit:               request.BaseCommit,
		EnvironmentSetupCommands: request.EnvironmentSetupCommands,
		Labels:                   map[string]string{"trajectory": request.TrajectoryID, "managed-by": "hostagent"},
	}
}

func buildErrorResponseMessage(request *model.RolloutRequest, err error) string {
	if err != nil {
		return fmt.Sprintf("The execution of the command %q failed, the error message is: %s", request.Commands, err.Error())
	}
	return ""
}

func determineExitCode(err error) int {
	if err != nil {
		return model.COMMAND_EXECUTION_ERROR
	}
	return model.COMMAND_EXECUTION_SUCCESS
}

func (m *Manager) Start() {
	m.logger.Info("Starting Docker Manager")
	m.CleanupAllContainers(context.Background())
	// Start a goroutine to handle incoming requests
	go func() {
		m.logger.Info("Listening for requests", zap.Int("queue_size", len(m.requestQueue)))
		fmt.Print(m.requestQueue == nil)
		for request := range m.requestQueue {
			go func(req model.RolloutRequest) {
				m.logger.Info("Received request", zap.String("ID", req.ID), zap.String("TrajectoryID", req.TrajectoryID))
				// Check if container exists for this trajectory
				containerID, exists := m.trajectoryInstanceMap[req.TrajectoryID]

				// If container doesn't exist, create one
				if !exists {
					m.logger.Info("Starting new container", zap.String("TrajectoryID", req.TrajectoryID))
					var err error
					containerID, err = m.StartContainer(context.Background(), req.ImageID, buildContainerOptions(&req))
					if err != nil {
						m.logger.Error("Failed to start container", zap.String("TrajectoryID", req.TrajectoryID), zap.Error(err))
						m.responseQueue <- model.RolloutResponse{
							ID:           req.ID,
							TrajectoryID: req.TrajectoryID,
							Error:        err.Error(),
							ExitCode:     model.INSTANCE_START_ERROR,
						}
						return
					}
					m.trajectoryInstanceMap[req.TrajectoryID] = containerID
				}

				// Run the command on the container (works for both new and existing containers)
				result, err := m.RunCommand(context.Background(), containerID, buildContainerOptions(&req), req.Timeout)
				result = utils.StripAnsi(result)
				m.responseQueue <- model.RolloutResponse{
					ID:           req.ID,
					TrajectoryID: req.TrajectoryID,
					Output:       result,
					Error:        buildErrorResponseMessage(&req, err),
					ExitCode:     determineExitCode(err),
				}
				if err != nil {
					m.logger.Error("Error running command", zap.Error(err))
				}
			}(request)
		}
	}()
}

// StartContainer starts a new Docker container with the given image and options
func (m *Manager) StartContainer(ctx context.Context, image string, options ContainerOptions) (string, error) {
	// Pull the image if it doesn't exist
	_, err := m.client.ImagePull(ctx, image, types.ImagePullOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to pull image %s: %w", image, err)
	}

	// Configure network mode
	networkMode := container.NetworkMode("bridge")
	if options.NetworkDisabled {
		networkMode = container.NetworkMode("none")
	}
	containerConfig := &container.Config{
		Image: image,
		Env:   []string{"TERM=xterm", "LC_ALL=C.UTF-8"},
		User:  options.User,
		// Cmd:             options.EnvironmentSetupCommands,
		Cmd:             []string{"/bin/bash"},
		Tty:             true,
		AttachStdin:     true,
		OpenStdin:       true,
		StdinOnce:       false,
		WorkingDir:      options.WorkingDir,
		Labels:          options.Labels,
		NetworkDisabled: options.NetworkDisabled,
	}

	hostConfig := &container.HostConfig{
		NetworkMode: networkMode,
	}

	resp, err := m.client.ContainerCreate(
		ctx,
		containerConfig,
		hostConfig,
		nil,
		nil,
		options.Name,
	)
	if err != nil {
		return "", fmt.Errorf("failed to create container: %w", err)
	}

	// Start the container
	if err := m.client.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return "", fmt.Errorf("failed to start container: %w", err)
	}

	return resp.ID, nil
}

// RunCommand executes a command in a running container and returns the output
func (m *Manager) RunCommand(ctx context.Context, containerID string, options ContainerOptions, timeout time.Duration) (string, error) {
	execConfig := types.ExecConfig{
		User:         options.User,
		WorkingDir:   options.WorkingDir,
		AttachStdout: true,
		AttachStderr: true,
		Cmd:          options.Command,
	}

	execResp, err := m.client.ContainerExecCreate(ctx, containerID, execConfig)
	if err != nil {
		return "", fmt.Errorf("failed to create exec: %w", err)
	}

	resp, err := m.client.ContainerExecAttach(ctx, execResp.ID, types.ExecStartCheck{})
	if err != nil {
		return "", fmt.Errorf("failed to attach to exec: %w", err)
	}
	defer resp.Close()

	// Read the output with timeout
	outputCh := make(chan []byte)
	errCh := make(chan error)
	go func() {
		output, err := io.ReadAll(resp.Reader)
		if err != nil {
			errCh <- err
			return
		}
		outputCh <- output
	}()

	var output []byte
	select {
	case output = <-outputCh:
		// got output
	case err := <-errCh:
		return "", fmt.Errorf("failed to read exec output: %w", err)
	case <-time.After(timeout):
		// Try to read whatever output is available so far
		select {
		case output = <-outputCh:
			return string(output), fmt.Errorf("command timed out after %s", timeout)
		default:
			return "", fmt.Errorf("command timed out after %s", timeout)
		}
	}

	// Check the exit code
	inspectResp, err := m.client.ContainerExecInspect(ctx, execResp.ID)
	if err != nil {
		return "", fmt.Errorf("failed to inspect exec: %w", err)
	}

	if inspectResp.ExitCode != 0 {
		return string(output), fmt.Errorf("command exited with code %d", inspectResp.ExitCode)
	}

	return string(output), nil
}

// GetContainerStats retrieves resource usage statistics for a container
func (m *Manager) GetContainerStats(ctx context.Context, containerID string) (*ContainerStats, error) {
	stats, err := m.client.ContainerStats(ctx, containerID, false)
	if err != nil {
		return nil, fmt.Errorf("failed to get container stats: %w", err)
	}
	defer stats.Body.Close()

	var statsJSON types.StatsJSON
	if err := json.NewDecoder(stats.Body).Decode(&statsJSON); err != nil {
		return nil, fmt.Errorf("failed to decode stats JSON: %w", err)
	}

	// Calculate CPU usage percentage
	cpuDelta := float64(statsJSON.CPUStats.CPUUsage.TotalUsage - statsJSON.PreCPUStats.CPUUsage.TotalUsage)
	systemDelta := float64(statsJSON.CPUStats.SystemUsage - statsJSON.PreCPUStats.SystemUsage)
	cpuUsage := 0.0
	if systemDelta > 0 && cpuDelta > 0 {
		cpuUsage = (cpuDelta / systemDelta) * float64(len(statsJSON.CPUStats.CPUUsage.PercpuUsage)) * 100.0
	}

	// Calculate memory usage
	memoryUsage := 0.0
	if statsJSON.MemoryStats.Limit > 0 {
		memoryUsage = float64(statsJSON.MemoryStats.Usage) / float64(statsJSON.MemoryStats.Limit) * 100.0
	}

	// Get container info for name
	containerInfo, err := m.client.ContainerInspect(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect container: %w", err)
	}

	name := strings.TrimPrefix(containerInfo.Name, "/")

	return &ContainerStats{
		ContainerID: containerID,
		Name:        name,
		CPUUsage:    cpuUsage,
		MemoryUsage: memoryUsage,
		MemoryUsed:  statsJSON.MemoryStats.Usage,
		MemoryLimit: statsJSON.MemoryStats.Limit,
		DiskRead:    statsJSON.BlkioStats.IoServiceBytesRecursive[0].Value,
		DiskWrite:   statsJSON.BlkioStats.IoServiceBytesRecursive[1].Value,
	}, nil
}

func (m *Manager) StartJanitor() (string, bool) {
	// Start a goroutine to clean up dead containers
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			ctx := context.Background()

			// Get all containers managed by this host agent
			containers, err := m.client.ContainerList(ctx, types.ContainerListOptions{
				All: true,
				Filters: filters.NewArgs(
					filters.Arg("label", "managed-by=hostagent"),
				),
			})

			if err != nil {
				m.logger.Error("Failed to list containers during janitor run", zap.Error(err))
				continue
			}

			for _, c := range containers {
				// Inspect container to check its status
				inspect, err := m.client.ContainerInspect(ctx, c.ID)
				if err != nil {
					m.logger.Error("Failed to inspect container", zap.String("containerID", c.ID), zap.Error(err))
					continue
				}

				// Check for containers that need cleanup
				needsCleanup := false
				cleanupReason := ""

				// Check if container is dead, exited, or OOM killed
				if !inspect.State.Running {
					needsCleanup = true
					cleanupReason = "container not running: " + inspect.State.Status
				} else if inspect.State.OOMKilled {
					needsCleanup = true
					cleanupReason = "container was OOM killed"
				} else if inspect.State.Health != nil && inspect.State.Health.Status == "unhealthy" {
					needsCleanup = true
					cleanupReason = "container is unhealthy"
				} else if inspect.State.Dead {
					needsCleanup = true
					cleanupReason = "container is dead"
				} else if inspect.State.Paused {
					needsCleanup = true
					cleanupReason = "container is paused"
				}

				// Clean up if necessary
				if needsCleanup {
					m.logger.Info("Cleaning up container",
						zap.String("containerID", c.ID),
						zap.String("name", c.Names[0]),
						zap.String("reason", cleanupReason))

					if err := m.CleanupContainer(ctx, c.ID); err != nil {
						m.logger.Error("Error cleaning up container",
							zap.String("containerID", c.ID),
							zap.Error(err))
					} else {
						// Remove from tracking map if it exists there
						for trajID, containerID := range m.trajectoryInstanceMap {
							if containerID == c.ID {
								delete(m.trajectoryInstanceMap, trajID)
								m.logger.Info("Removed container from trajectory map",
									zap.String("trajectoryID", trajID),
									zap.String("containerID", c.ID))
								break
							}
						}
					}
				}
			}
		}
	}()

	return "Janitor started - monitoring for dead containers every 10 seconds", true
}

// func (m *Manager) GetPatch() (string, error) {
// 	command := fmt.Sprintf("bash -c 'git diff %s'", baseCommit)
// 	execConfig := types.ExecConfig{
// 		Cmd:          []string{"/bin/bash", "-c", command},
// 		AttachStdout: true,
// 		AttachStderr: true,
// 	}
// 	execID, err := m.client.ContainerExecCreate(context.Background(), m.containerID, execConfig)
// 	if err != nil {
// 		return "", fmt.Errorf("failed to create exec instance: %w", err)
// 	}
// 	resp, err := m.client.ContainerExecAttach(context.Background(), execID.ID, types.ExecStartCheck{})
// 	if err != nil {
// 		return "", fmt.Errorf("failed to attach to exec instance: %w", err)
// 	}
// 	defer resp.Close()
// 	var outputBuffer bytes.Buffer
// 	_, err = io.Copy(&outputBuffer, resp.Reader)
// 	if err != nil {
// 		return "", fmt.Errorf("failed to read exec output: %w", err)
// 	}
// 	output := outputBuffer.String()
// 	// Strip ANSI codes if needed
// 	// output = t.StripAnsi(output)
// 	return output, nil
// }

// CleanupContainer stops and removes a container
func (m *Manager) CleanupContainer(ctx context.Context, containerID string) error {
	// Stop the container
	timeout := int(0) // 30 seconds timeout
	if err := m.client.ContainerStop(ctx, containerID, container.StopOptions{Timeout: &timeout, Signal: "SIGKILL"}); err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}

	// Remove the container
	if err := m.client.ContainerRemove(ctx, containerID, types.ContainerRemoveOptions{
		RemoveVolumes: true,
		Force:         true,
	}); err != nil {
		return fmt.Errorf("failed to remove container: %w", err)
	}

	// Remove the container's cache
	if _, err := m.client.ContainersPrune(ctx, filters.NewArgs(filters.Arg("id", containerID))); err != nil {
		return fmt.Errorf("failed to remove container cache: %w", err)
	}

	return nil
}

// CleanupAllContainers stops and removes all containers managed by this agent
func (m *Manager) CleanupAllContainers(ctx context.Context) error {
	// Get all containers with our label
	containers, err := m.client.ContainerList(ctx, types.ContainerListOptions{
		All: true,
		Filters: filters.NewArgs(
			filters.Arg("label", "managed-by=hostagent"),
		),
	})
	m.logger.Info("Cleaning up all containers", zap.Int("count", len(containers)))
	if err != nil {
		return fmt.Errorf("failed to list containers: %w", err)
	}

	for _, c := range containers {
		if err := m.CleanupContainer(ctx, c.ID); err != nil {
			// Log error but continue with other containers
			fmt.Printf("Error cleaning up container %s: %v\n", c.ID, err)
		}
	}

	return nil
}

package docker

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/google/uuid"
	"github.com/multiturn-rl-hostagent/model"
	"go.uber.org/zap"
)

// InstanceDetails represents options for creating a container
type InstanceDetails struct {
	ContainerID             string
	ContainerName           string
	ImageID                 string
	Env                     []string
	User                    string
	WorkingDir              string
	NetworkDisabled         bool
	BaseCommit              string
	EnvironmentSetupCommand string
	Labels                  map[string]string
}

// ContainerStats represents resource usage statistics for a container
type ContainerStats struct {
	ContainerID string
	Name        string
	CPUUsage    float64 // percentage
	MemoryUsage float64 // percentage
	MemoryUsed  uint64  // bytes
	MemoryLimit uint64  // bytes
}

type ContainerShell struct {
	conn   net.Conn
	reader *bufio.Reader
	writer io.Writer
	Marker string
}

// Manager handles Docker container operations
type Manager struct {
	client                *client.Client
	sessions              map[string]*ContainerShell
	logger                *zap.Logger
	requestQueue          chan model.RolloutRequest
	responseQueue         chan model.RolloutResponse
	trajectoryInstanceMap map[string]InstanceDetails // Map of trajectory ID to instance details
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
		sessions:              make(map[string]*ContainerShell),
		logger:                logger,
		requestQueue:          requestQueue,
		responseQueue:         responseQueue,
		trajectoryInstanceMap: make(map[string]InstanceDetails),
	}, nil
}

func buildInstanceDetails(request *model.RolloutRequest) InstanceDetails {
	return InstanceDetails{
		ImageID:                 request.ImageID,
		User:                    request.User,
		ContainerName:           request.TrajectoryID,
		WorkingDir:              request.WorkingDir,
		NetworkDisabled:         request.NetworkDisabled,
		BaseCommit:              request.BaseCommit,
		EnvironmentSetupCommand: request.EnvironmentSetupCommand,
		Labels:                  map[string]string{"trajectory": request.TrajectoryID, "managed-by": "hostagent"},
	}
}

func buildErrorResponseMessage(request *model.RolloutRequest, err error) string {
	if err != nil {
		return fmt.Sprintf("The execution of the command %q failed, the error message is: %s", request.Command, err.Error())
	}
	return ""
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
				instanceDetails, exists := m.trajectoryInstanceMap[req.TrajectoryID]

				// If container doesn't exist, create one
				if !exists {
					m.logger.Info("Starting new container", zap.String("TrajectoryID", req.TrajectoryID))
					var err error
					instanceDetails = buildInstanceDetails(&req)
					_, err = m.StartContainer(context.Background(), &instanceDetails)
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
					m.trajectoryInstanceMap[req.TrajectoryID] = instanceDetails
				}

				// if req.RequestType == model.REQUEST_TYPE_GET_PATCH {
				// 	m.logger.Info("Getting patch", zap.String("TrajectoryID", req.TrajectoryID))
				// 	patch, returnCode, err := m.GetPatch(&instanceDetails)
				// 	if err != nil {
				// 		m.logger.Error("Failed to get patch", zap.String("TrajectoryID", req.TrajectoryID), zap.Error(err))
				// 	}
				// 	m.responseQueue <- model.RolloutResponse{
				// 		ID:           req.ID,
				// 		TrajectoryID: req.TrajectoryID,
				// 		Patch:        patch,
				// 		Error:        buildErrorResponseMessage(&req, err),
				// 		ReturnReason: returnCode,
				// 	}
				// 	m.CleanupTrajectory(req.TrajectoryID)
				// 	return // This is the end of a single trajectory request
				// }

				if req.RequestType == model.REQUEST_TYPE_GET_OUTPUT {
					m.logger.Info("Getting output", zap.String("TrajectoryID", req.TrajectoryID))
					output, err := m.GetOutput(&instanceDetails)
					if err != nil {
						m.logger.Error("Failed to get output", zap.String("TrajectoryID", req.TrajectoryID), zap.Error(err))
						m.responseQueue <- model.RolloutResponse{
							ID:           req.ID,
							TrajectoryID: req.TrajectoryID,
							Error:        buildErrorResponseMessage(&req, err),
						}
						return
					}
					m.responseQueue <- model.RolloutResponse{
						ID:           req.ID,
						TrajectoryID: req.TrajectoryID,
						Output:       output,
					}
					return
				}

				// Run the command on the container (works for both new and existing containers)
				err := m.sessions[instanceDetails.ContainerID].Execute(req.Command, req.TrajectoryID)
				// m.responseQueue <- model.RolloutResponse{
				// 	ID:           req.ID,
				// 	TrajectoryID: req.TrajectoryID,
				// 	Output:       result,
				// 	Error:        buildErrorResponseMessage(&req, err),
				// 	ExitCode:     returnCode,
				// }
				if err != nil {
					m.logger.Error("Error running command", zap.Error(err))
				}
			}(request)
		}
	}()
}

func (m *Manager) StartContainer(ctx context.Context, instanceDetails *InstanceDetails) (string, error) {
	// Pull image
	_, err := m.client.ImagePull(ctx, instanceDetails.ImageID, types.ImagePullOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to pull image %s: %w", instanceDetails.ImageID, err)
	}

	// Container config
	networkMode := container.NetworkMode("bridge")
	if instanceDetails.NetworkDisabled {
		networkMode = container.NetworkMode("none")
	}

	containerConfig := &container.Config{
		Image:           instanceDetails.ImageID,
		Env:             []string{"TERM=xterm", "LC_ALL=C.UTF-8"},
		User:            instanceDetails.User,
		Entrypoint:      []string{"/bin/bash"},
		WorkingDir:      instanceDetails.WorkingDir,
		Labels:          instanceDetails.Labels,
		NetworkDisabled: instanceDetails.NetworkDisabled,
		Tty:             true,
		OpenStdin:       true,
		AttachStdin:     true,
		AttachStdout:    true,
		AttachStderr:    true,
	}

	hostConfig := &container.HostConfig{
		NetworkMode: networkMode,
	}

	resp, err := m.client.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, instanceDetails.ContainerName)
	if err != nil {
		return "", fmt.Errorf("failed to create container: %w", err)
	}

	if err := m.client.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return "", fmt.Errorf("failed to start container: %w", err)
	}

	attachResp, err := m.client.ContainerAttach(ctx, resp.ID, types.ContainerAttachOptions{
		Stream: true, Stdin: true, Stdout: true, Stderr: true,
	})
	if err != nil {
		return "", fmt.Errorf("failed to attach to container: %w", err)
	}

	// Wrap the attach session as persistent shell
	session := &ContainerShell{
		conn:   attachResp.Conn,
		reader: bufio.NewReader(attachResp.Reader),
		writer: attachResp.Conn,
		Marker: fmt.Sprintf("__CMD_DONE__%s__", uuid.New().String()),
	}
	instanceDetails.ContainerID = resp.ID
	f, err := os.Create(getInstanceLogFilePath(instanceDetails))
	go func() {
		defer func() {
			f.Close()
			fmt.Printf("Closed file for container %s\n", instanceDetails.ContainerID)
		}()
		_, err := io.Copy(f, attachResp.Reader)
		if err != nil {
			log.Printf("Error copying output: %v\n", err)
		}
	}()

	m.sessions[instanceDetails.ContainerID] = session
	m.logger.Info("Container started and shell session attached", zap.String("containerID", resp.ID))

	return resp.ID, nil
}

func getInstanceLogFilePath(instanceDetails *InstanceDetails) string {
	return fmt.Sprintf("./tmp/container-output-trajectory-%s.txt", instanceDetails.ContainerName)
}

func EncodeInput(text string) []byte {
	if strings.HasPrefix(text, "^") {
		var result []byte
		i := 0
		for i < len(text) {
			if text[i] == '^' {
				i++
				if i >= len(text) {
					break
				}
				char := text[i]
				if char >= 64 && char <= 127 {
					result = append(result, char-64) // Ctrl+X => ASCII(X) - 64
				}
			} else {
				result = append(result, text[i])
			}
			i++
		}
		return result
	}
	return []byte(text)
}

func (s *ContainerShell) Execute(cmd, trajectoryID string) error {
	marker := s.Marker

	fullCmd := fmt.Sprintf("%s ; echo %s", cmd, marker)

	if _, err := s.writer.Write(EncodeInput(fullCmd + "\n")); err != nil {
		log.Printf("Error writing command: %v\n", err)
		return fmt.Errorf("failed to write command: %w", err)
	}
	return nil
}

func (m *Manager) GetOutput(instanceDetails *InstanceDetails) (string, error) {
	logFilePath := getInstanceLogFilePath(instanceDetails)
	output, err := os.ReadFile(logFilePath)
	if err != nil {
		return "", fmt.Errorf("failed to read output file: %w", err)
	}
	cleanOutput := CleanUseEmulator(output)
	marker := m.sessions[instanceDetails.ContainerID].Marker
	// Find the last occurrence of the marker command
	lastCmdStartIndex := strings.LastIndex(cleanOutput, fmt.Sprintf("; echo %s", marker))
	if lastCmdStartIndex != -1 {
		cleanOutput = cleanOutput[lastCmdStartIndex+len(fmt.Sprintf("; echo %s", marker))+1:]
	}
	// Remove the marker command from the output
	cleanOutput = strings.Replace(cleanOutput, fmt.Sprintf("%s\n", marker), "", 1)
	return cleanOutput, nil
}

func (m *Manager) GetPatch(instanceDetails *InstanceDetails) error {
	command := fmt.Sprintf("bash -c 'git diff %s'", instanceDetails.BaseCommit)
	return m.sessions[instanceDetails.ContainerID].Execute(command, instanceDetails.ContainerName)
}

func (m *Manager) CleanupTrajectory(trajectoryID string) error {
	if instance, exists := m.trajectoryInstanceMap[trajectoryID]; exists {
		return m.CleanupContainer(context.Background(), instance.ContainerID)
	}
	return nil
}

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
	m.client.ContainersPrune(ctx, filters.NewArgs(filters.Arg("id", containerID)))
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

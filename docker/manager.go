package docker

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/google/uuid"
	"github.com/multiturn-rl-hostagent/model"
	"github.com/multiturn-rl-hostagent/utils"
	"go.uber.org/zap"
)

// InstanceDetails represents options for creating a container
type InstanceDetails struct {
	ContainerID           string
	TrajectoryID          string
	ImageID               string
	Env                   []string
	ShellPath             string
	User                  string
	WorkingDir            string
	NetworkDisabled       bool
	Labels                map[string]string
	LastestOutputPosition int
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
	requestQueue          chan model.RolloutRequestInput
	responseQueue         chan model.RolloutResponse
	trajectoryInstanceMap map[string]*InstanceDetails // Map of trajectory ID to instance details
}

// NewManager creates a new Docker manager
func NewManager(requestQueue chan model.RolloutRequestInput, responseQueue chan model.RolloutResponse) (*Manager, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	return &Manager{
		client:                cli,
		sessions:              make(map[string]*ContainerShell),
		logger:                utils.GetLogger(),
		requestQueue:          requestQueue,
		responseQueue:         responseQueue,
		trajectoryInstanceMap: make(map[string]*InstanceDetails),
	}, nil
}

func buildInstanceDetails(request *model.RolloutRequestInput) *InstanceDetails {
	return &InstanceDetails{
		ImageID:               request.StartSandboxInput.ImageID,
		User:                  request.StartSandboxInput.User,
		TrajectoryID:          request.TrajectoryID,
		WorkingDir:            request.StartSandboxInput.WorkingDir,
		ShellPath:             request.StartSandboxInput.ShellPath,
		NetworkDisabled:       request.StartSandboxInput.NetworkDisabled,
		Labels:                map[string]string{"trajectory": request.TrajectoryID, "managed-by": "hostagent"},
		LastestOutputPosition: 0,
	}
}

func buildErrorResponseMessage(err error) string {
	if err != nil {
		return fmt.Sprintf("Sandbox encounter an error, the error message is: %s", err.Error())
	}
	return ""
}

func (m *Manager) Start() {
	m.CleanupAllContainers(context.Background())
	// // Start a goroutine to handle incoming requests
	// go func() {
	// 	m.logger.Info("Listening for requests", zap.Int("queue_size", len(m.requestQueue)))
	// 	for request := range m.requestQueue {
	// 		go func(req model.RolloutRequestInput) {
	// 			m.logger.Info("Received request", zap.String("ID", req.ID), zap.String("TrajectoryID", req.TrajectoryID))
	// 			switch req.RequestType {
	// 			case model.REQUEST_TYPE_RUN_COMMAND:
	// 				m.logger.Info("Running command", zap.String("TrajectoryID", req.TrajectoryID), zap.String("Command", req.RunCommandInput.Command))
	// 				m.handleRunCommand(req)
	// 			default:
	// 				m.logger.Error("Unknown request type", zap.Uint8("RequestType", req.RequestType))
	// 			}
	// 		}(request)
	// 	}
	// }()
	m.logger.Info("Docker Manager started")
}

func (m *Manager) HandleStartSandbox(req model.RolloutRequestInput) error {
	m.logger.Info("Starting new container", zap.String("TrajectoryID", req.TrajectoryID))
	var err error
	instanceDetails := buildInstanceDetails(&req)
	_, err = m.StartContainer(context.Background(), instanceDetails)
	if err != nil {
		m.logger.Error("Failed to start container", zap.String("TrajectoryID", req.TrajectoryID), zap.Error(err))
		m.responseQueue <- model.RolloutResponse{
			ID:           req.ID,
			TrajectoryID: req.TrajectoryID,
			Error:        err.Error(),
			ExitCode:     model.INSTANCE_START_ERROR,
		}
		return fmt.Errorf("failed to start container: %w", err)
	}
	m.trajectoryInstanceMap[req.TrajectoryID] = instanceDetails
	return nil
}

func (m *Manager) HandleShutdownSandbox(req model.RolloutRequestInput) {
	m.logger.Info("Shutting down container", zap.String("TrajectoryID", req.TrajectoryID))
	m.CleanupTrajectory(req.TrajectoryID)
}

func (m *Manager) HandleRunCommand(req model.RolloutRequestInput) (string, error) {
	m.logger.Info("Running command", zap.String("TrajectoryID", req.TrajectoryID))
	instanceDetails, exists := m.trajectoryInstanceMap[req.TrajectoryID]
	if !exists {
		m.logger.Error("Instance not found", zap.String("TrajectoryID", req.TrajectoryID))
		m.responseQueue <- model.RolloutResponse{
			ID:           req.ID,
			TrajectoryID: req.TrajectoryID,
			Error:        "Instance not found",
			ExitCode:     model.INTERNAL_ERROR,
		}
		return "", fmt.Errorf("instance not found")
	}

	if req.RunCommandInput.IsInteractive {
		m.sessions[instanceDetails.ContainerID].Execute(req.RunCommandInput.Command, req.TrajectoryID)
		time.Sleep(time.Duration(req.RunCommandInput.TimeoutInSeconds) * time.Second)
		return m.handleGetOutput(req, true)
	} else {
		result, err := m.StartExecRunCommand(req.RunCommandInput.Command, instanceDetails, instanceDetails.User, instanceDetails.WorkingDir, nil, false)
		if err != nil {
			m.logger.Error("Failed to run command", zap.String("TrajectoryID", req.TrajectoryID), zap.Error(err))
			m.responseQueue <- model.RolloutResponse{
				ID:           req.ID,
				TrajectoryID: req.TrajectoryID,
				Error:        buildErrorResponseMessage(err),
			}
			return "", err
		}
		// m.responseQueue <- model.RolloutResponse{
		// 	ID:           req.ID,
		// 	TrajectoryID: req.TrajectoryID,
		// 	Output:       string(result.Output),
		// 	ExitCode:     result.ExitCode,
		// }
		return string(result.Output), nil
	}
}

func (m *Manager) handleGetOutput(req model.RolloutRequestInput, async bool) (string, error) {
	m.logger.Info("Getting output", zap.String("TrajectoryID", req.TrajectoryID))
	instanceDetails, exists := m.trajectoryInstanceMap[req.TrajectoryID]
	if !exists {
		m.logger.Error("Instance not found", zap.String("TrajectoryID", req.TrajectoryID))
		m.responseQueue <- model.RolloutResponse{
			ID:           req.ID,
			TrajectoryID: req.TrajectoryID,
			Error:        "Instance not found",
			ExitCode:     model.INTERNAL_ERROR,
		}
		return "", fmt.Errorf("instance not found")
	}
	output, _, err := m.GetOutput(instanceDetails.TrajectoryID)
	if err != nil {
		m.logger.Error("Failed to get output", zap.String("TrajectoryID", req.TrajectoryID), zap.Error(err))
		m.responseQueue <- model.RolloutResponse{
			ID:           req.ID,
			TrajectoryID: req.TrajectoryID,
			Error:        buildErrorResponseMessage(err),
		}
		return "", err
	}
	if async {
		m.responseQueue <- model.RolloutResponse{
			ID:           req.ID,
			TrajectoryID: req.TrajectoryID,
			Output:       output,
		}
		return output, nil
	}
	return output, nil
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
		Image: instanceDetails.ImageID,
		Env:   []string{"TERM=xterm", "LC_ALL=C.UTF-8"},
		User:  instanceDetails.User,
		Entrypoint: func() []string {
			if instanceDetails.ShellPath == "" {
				return []string{"/bin/bash"}
			}
			return []string{instanceDetails.ShellPath}
		}(),
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

	resp, err := m.client.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, instanceDetails.TrajectoryID)
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
			utils.GetLogger().Error("Error copying output", zap.Error(err))
		}
	}()

	m.sessions[instanceDetails.ContainerID] = session
	m.logger.Info("Container started and shell session attached", zap.String("containerID", resp.ID))

	return resp.ID, nil
}

type ExecResult struct {
	ExitCode int
	Output   []byte
}

// StartExecRunCommand runs a command inside a container, similar to docker exec.
// Only uses cmd, user, and workdir from the arguments.
func (m *Manager) StartExecRunCommand(
	cmd interface{},
	instanceDetails *InstanceDetails,
	user string,
	workdir string,
	env []string,
	privileged bool,
) (ExecResult, error) {
	ctx := context.Background()
	containerID := instanceDetails.ContainerID

	// Prepare command
	var cmdArr []string
	switch v := cmd.(type) {
	case string:
		cmdArr = []string{"/bin/sh", "-c", v}
	case []string:
		cmdArr = v
	default:
		return ExecResult{}, fmt.Errorf("cmd must be string or []string")
	}

	execConfig := types.ExecConfig{
		Cmd:          cmdArr,
		User:         user,
		WorkingDir:   workdir,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
		Env:          env,
		Privileged:   privileged,
	}

	resp, err := m.client.ContainerExecCreate(ctx, containerID, execConfig)
	if err != nil {
		return ExecResult{}, fmt.Errorf("failed to create exec instance: %w", err)
	}

	attachResp, err := m.client.ContainerExecAttach(ctx, resp.ID, types.ExecStartCheck{Tty: false})
	if err != nil {
		return ExecResult{}, fmt.Errorf("failed to attach to exec instance: %w", err)
	}
	defer attachResp.Close()

	output, err := io.ReadAll(attachResp.Reader)
	if err != nil {
		return ExecResult{}, fmt.Errorf("failed to read output: %w", err)
	}

	inspectResp, err := m.client.ContainerExecInspect(ctx, resp.ID)
	if err != nil {
		return ExecResult{}, fmt.Errorf("failed to inspect exec instance: %w", err)
	}

	return ExecResult{ExitCode: inspectResp.ExitCode, Output: output}, nil
}

func getInstanceLogFilePath(instanceDetails *InstanceDetails) string {
	return fmt.Sprintf("./tmp/container-output-trajectory-%s.txt", instanceDetails.TrajectoryID)
}

func EncodeInput(text string) []byte {
	if len(text) == 2 && strings.HasPrefix(text, "^") {
		ctrl := text[1]
		if ctrl >= 64 && ctrl <= 95 {
			return []byte{ctrl - 64} // Ctrl+X => ASCII(X) - 64
		}
	}

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
				result = append(result, char-64)
			}
		} else {
			result = append(result, text[i])
		}
		i++
	}
	return result
}

var UserMarker = false

func (s *ContainerShell) Execute(cmd, trajectoryID string) error {
	marker := s.Marker

	// Handle control character commands
	if len(cmd) == 2 && strings.HasPrefix(cmd, "^") {
		if _, err := s.writer.Write(EncodeInput(cmd)); err != nil {
			utils.GetLogger().Error("Error writing control command", zap.Error(err))
			return fmt.Errorf("failed to write control command: %w", err)
		}
		return nil
	}

	// Determine full command based on whether marker is used
	var fullCmd string
	if UserMarker {
		fullCmd = fmt.Sprintf("%s ; echo %s", cmd, marker)
	} else {
		fullCmd = cmd
	}

	// Write the command to the container
	if _, err := s.writer.Write(EncodeInput(fullCmd + "\n")); err != nil {
		utils.GetLogger().Error("Error writing command", zap.Error(err))
		return fmt.Errorf("failed to write command: %w", err)
	}
	return nil
}

func (m *Manager) GetOutput(trajectoryID string) (string, bool, error) {
	instanceDetails, exists := m.trajectoryInstanceMap[trajectoryID]
	if !exists {
		return "", false, fmt.Errorf("instance not found")
	}

	logFilePath := getInstanceLogFilePath(instanceDetails)
	output, err := os.ReadFile(logFilePath)
	if err != nil {
		return "", true, fmt.Errorf("failed to read output file: %w", err)
	}

	cleanOutputAll := CleanUseEmulator(output)
	cleanOutput := cleanOutputAll[instanceDetails.LastestOutputPosition:]
	instanceDetails.LastestOutputPosition = len(cleanOutputAll)
	m.logger.Debug("Output position updated", zap.Int("position", instanceDetails.LastestOutputPosition), zap.String("trajectoryID", trajectoryID))
	commandFinished := true

	// Handle marker-based output processing if enabled
	if UserMarker {
		marker := m.sessions[instanceDetails.ContainerID].Marker

		// Find the last occurrence of the marker command
		lastCmdStartIndex := strings.LastIndex(cleanOutput, fmt.Sprintf("; echo %s", marker))
		if lastCmdStartIndex != -1 {
			cleanOutput = cleanOutput[lastCmdStartIndex+len(fmt.Sprintf("; echo %s", marker))+1:]
		}

		// Check if command finished by looking for marker
		commandFinished = strings.Contains(cleanOutput, marker)

		// Remove the markers from the output
		cleanOutput = strings.ReplaceAll(cleanOutput, fmt.Sprintf("%s\n", marker), "")
	}

	return cleanOutput, commandFinished, nil
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
	timeout := int(2) // 30 seconds timeout
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

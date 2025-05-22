package model

import "time"

type RolloutRequest struct {
	ID                       string        `json:"id"`                                   // Unique identifier for the rollout request
	TrajectoryID             string        `json:"trajectory"`                           // Associated trajectory ID
	ImageID                  string        `json:"image_id"`                             // Docker image ID
	Commands                 []string      `json:"commands,omitempty"`                   // Commands to execute in the container
	Timeout                  time.Duration `json:"timeout,omitempty"`                    // Command execution timeout
	User                     string        `json:"user,omitempty"`                       // User to run commands as
	WorkingDir               string        `json:"working_dir,omitempty"`                // Working directory inside the container
	NetworkDisabled          bool          `json:"network_disabled,omitempty"`           // Container network mode
	RequestType              uint8         `json:"request_type,omitempty"`               // Type of request (e.g., 0 for rollout, 1 for get_patch)
	BaseCommit               string        `json:"base_commit,omitempty"`                // Base commit for the rollout
	EnvironmentSetupCommands []string      `json:"environment_setup_commands,omitempty"` // Commands to set up the environment
}

type RolloutResponse struct {
	ID             string `json:"id"`              // Unique identifier for the rollout request
	TrajectoryID   string `json:"trajectory"`      // Associated trajectory ID
	InstanceStatus string `json:"instance_status"` // Status of the container instance
	Output         string `json:"output"`          // Command output
	Patch          string `json:"patch"`           // Patch generated from the command output
	Error          string `json:"error,omitempty"` // Error message, if any
	ExitCode       int    `json:"exit_code"`       // Command exit code
}

const (
	COMMAND_EXECUTION_ERROR   = 400
	COMMAND_EXECUTION_SUCCESS = 200
	COMMAND_EXECUTION_TIMEOUT = 408

	INSTANCE_START_ERROR = 500
)

package model

type RolloutRequest struct {
	ID                      string `json:"id"`                                   // Unique identifier for the rollout request
	TrajectoryID            string `json:"trajectory"`                           // Associated trajectory ID
	ImageID                 string `json:"image_id"`                             // Docker image ID
	Command                 string `json:"commands,omitempty"`                   // Commands to execute in the container
	TimeoutInSeconds        int    `json:"timeout_in_seconds,omitempty"`         // Command execution timeout
	User                    string `json:"user,omitempty"`                       // User to run commands as
	WorkingDir              string `json:"working_dir,omitempty"`                // Working directory inside the container
	NetworkDisabled         bool   `json:"network_disabled,omitempty"`           // Container network mode
	RequestType             uint8  `json:"request_type,omitempty"`               // Type of request (e.g., 0 for rollout, 1 for get_patch)
	BaseCommit              string `json:"base_commit,omitempty"`                // Base commit for the rollout
	EnvironmentSetupCommand string `json:"environment_setup_commands,omitempty"` // Commands to set up the environment
}

type RolloutResponse struct {
	ID             string `json:"id"`              // Unique identifier for the rollout request
	TrajectoryID   string `json:"trajectory"`      // Associated trajectory ID
	InstanceStatus string `json:"instance_status"` // Status of the container instance
	Output         string `json:"output"`          // Command output
	Patch          string `json:"patch"`           // Patch generated from the command output
	Error          string `json:"error,omitempty"` // Error message, if any
	ExitCode       int    `json:"exit_code"`       // Command exit code
	ReturnReason   int    `json:"return_reason"`   // Reason for the command return
}

const (
	COMMAND_EXECUTION_ERROR   = 400
	COMMAND_EXECUTION_FINISH  = 200
	COMMAND_EXECUTION_TIMEOUT = 408

	INSTANCE_START_ERROR = 500
	INTERNAL_ERROR       = 500
)

const (
	REQUEST_TYPE_RUN_COMMAND = iota
	REQUEST_TYPE_GET_PATCH
	REQUEST_TYPE_GET_OUTPUT
)

package model

type RolloutRequestInput struct {
	ID                string            `json:"id"`                            // Unique identifier for the rollout request
	TrajectoryID      string            `json:"trajectory"`                    // Associated trajectory ID, used also for container name
	RequestType       uint8             `json:"request_type,omitempty"`        // Type of request
	StartSandboxInput StartSandboxInput `json:"start_sandbox_input,omitempty"` // Input for starting a sandbox
	RunCommandInput   RunCommandInput   `json:"run_command_input,omitempty"`   // Input for running a command
}

type StartSandboxInput struct {
	ImageID         string `json:"image_id"`                   // Docker image ID
	User            string `json:"user"`                       // User to run commands as
	WorkingDir      string `json:"working_dir,omitempty"`      // Working directory inside the container
	NetworkDisabled bool   `json:"network_disabled,omitempty"` // Container network mode
	ShellPath       string `json:"shell_path,omitempty"`       // Path to the shell executable
}

type RunCommandInput struct {
	Command          string   `json:"command"`
	Env              []string `json:"env,omitempty"`                // Environment variables to set for the command
	TimeoutInSeconds int      `json:"timeout_in_seconds,omitempty"` // Timeout for the command execution (seconds)
	WorkingDir       string   `json:"working_dir,omitempty"`        // Working directory for the command
	NetworkDisabled  bool     `json:"network_disabled,omitempty"`   // Whether to disable network access for the command
	ShellPath        string   `json:"shell_path,omitempty"`         // Path to the shell executable
	IsInteractive    bool     `json:"is_interactive,omitempty"`     // Whether the command is interactive
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
	REQUEST_TYPE_GET_OUTPUT
	REQUEST_TYPE_START_SANDBOX
	REQUEST_TYPE_SHUTDOWN_SANDBOX
)

const (
	SANDBOX_ENGINE_INTERNAL_ERROR = iota
)

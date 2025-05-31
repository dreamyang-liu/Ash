package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/multiturn-rl-hostagent/docker"
	"github.com/multiturn-rl-hostagent/model"
	"github.com/multiturn-rl-hostagent/utils"
	"go.uber.org/zap"
)

// HostAgent represents the main agent that manages Docker containers and monitors resources
type HostAgent struct {
	dockerManager *docker.Manager
	requestQueue  chan model.RolloutRequestInput
	responseQueue chan model.RolloutResponse
	ctx           context.Context
	cancel        context.CancelFunc
}

// NewHostAgent creates a new instance of HostAgent
func NewHostAgent() (*HostAgent, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Initialize request and response queues
	requestQueue := make(chan model.RolloutRequestInput, 50)
	responseQueue := make(chan model.RolloutResponse, 50)

	// Initialize Docker manager
	dockerManager, err := docker.NewManager(requestQueue, responseQueue)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to initialize Docker manager: %w", err)
	}

	// // Initialize RabbitMQ client
	// queueClient, err := queue.NewRabbitMQClient(os.Getenv("RABBITMQ_URL"))
	// if err != nil {
	// 	cancel()
	// 	return nil, fmt.Errorf("failed to initialize RabbitMQ client: %w", err)
	// }

	return &HostAgent{
		dockerManager: dockerManager,
		requestQueue:  requestQueue,
		responseQueue: responseQueue,
		ctx:           ctx,
		cancel:        cancel,
	}, nil
}

// InitHTTPServer sets up HTTP endpoints for the host agent
func (ha *HostAgent) InitHTTPServer(addr string) error {
	// Initialize host monitor if not already done

	http.HandleFunc("/start_sandbox", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req model.RolloutRequestInput
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		err := ha.dockerManager.HandleStartSandbox(req)
		if err != nil {
			utils.GetLogger().Error("Failed to start sandbox", zap.Error(err))
			http.Error(w, fmt.Sprintf("Failed to start sandbox: %v", err), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "sandbox creation initiated"})
	})

	http.HandleFunc("/shutdown_sandbox", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req model.RolloutRequestInput
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		go func() {
			ha.dockerManager.HandleShutdownSandbox(req)
		}()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "sandbox shutdown initiated"})
	})

	http.HandleFunc("/run_command", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req model.RolloutRequestInput
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		output, err := ha.dockerManager.HandleRunCommand(req)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to run command: %v", err), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "command execution initiated", "output": output})
	})

	http.HandleFunc("/get_output", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		trajectoryID := r.URL.Query().Get("trajectory_id")
		id := r.URL.Query().Get("id")

		if trajectoryID == "" || id == "" {
			http.Error(w, "Missing required parameters: trajectory_id and id", http.StatusBadRequest)
			return
		}

		output, _, _ := ha.dockerManager.GetOutput(trajectoryID)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(output)
	})

	// Start HTTP server in a goroutine
	go func() {
		utils.GetLogger().Info("HTTP server starting", zap.String("addr", addr))
		if err := http.ListenAndServe(addr, nil); err != nil {
			utils.GetLogger().Fatal("HTTP server failed", zap.Error(err))
		}
	}()

	return nil
}

// Start begins the host agent operations
func (ha *HostAgent) Start() error {
	utils.GetLogger().Info("Starting submodules ...")

	go ha.dockerManager.Start()

	// Wait for termination signal
	// sigCh := make(chan os.Signal, 1)
	// signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	// <-sigCh

	// utils.GetLogger().Info("Shutting down Host Agent...")
	// ha.Shutdown()
	return nil
}

// Shutdown stops all agent operations
func (ha *HostAgent) Shutdown() {
	ha.cancel()

	// Cleanup any running containers
	if ha.dockerManager != nil {
		ha.dockerManager.CleanupAllContainers(ha.ctx)
	}

	utils.GetLogger().Info("Host Agent shutdown complete")
}

func (ha *HostAgent) PutRequestToQueue(request model.RolloutRequestInput) {
	ha.requestQueue <- request
}

func (ha *HostAgent) GetResponseFromQueue() model.RolloutResponse {
	response := <-ha.responseQueue
	return response
}

func Initialize() {
	os.RemoveAll("./tmp")
	os.Mkdir("./tmp", 0755)
}

func main() {
	Initialize()
	utils.GetLogger().Info("Initializing host agent...")
	agent, err := NewHostAgent()
	if err != nil {
		log.Fatalf("Failed to initialize host agent: %v", err)
	}
	utils.GetLogger().Info("Host agent initialized successfully")

	// Start the agent in a goroutine so we can continue execution
	if err := agent.Start(); err != nil {
		log.Fatalf("Host agent error: %v", err)
	}

	agent.InitHTTPServer(":8080")

	go func() {
		for {
			response := agent.GetResponseFromQueue()
			writeResponseToFile(response)
		}
	}()

	// request := model.RolloutRequestInput{
	// 	ID:               "1233",
	// 	TrajectoryID:     "test-trajectory",
	// 	ImageID:          "ubuntu:latest",
	// 	Command:          "apt-get -y update && apt-get install -y git",
	// 	User:             "root",
	// 	WorkingDir:       "/testbed",
	// 	ShellPath:        "/bin/bash",
	// 	NetworkDisabled:  false,
	// 	TimeoutInSeconds: 5,
	// 	RequestType:      model.REQUEST_TYPE_RUN_COMMAND,
	// }

	// log.Println("Sending request to queue...")
	// agent.PutRequestToQueue(request)
	// time.Sleep(6 * time.Second)
	// request = model.RolloutRequestInput{
	// 	ID:               "1234",
	// 	TrajectoryID:     "test-trajectory",
	// 	ImageID:          "ubuntu:latest",
	// 	Command:          "^C",
	// 	User:             "root",
	// 	WorkingDir:       "/testbed",
	// 	NetworkDisabled:  false,
	// 	TimeoutInSeconds: 5,
	// 	RequestType:      model.REQUEST_TYPE_RUN_COMMAND,
	// }

	// log.Println("Sending request to queue...")
	// agent.PutRequestToQueue(request)

	// time.Sleep(6 * time.Second)
	// // time.Sleep(30 * time.Second)
	// request = model.RolloutRequestInput{
	// 	ID:               "1235",
	// 	TrajectoryID:     "test-trajectory",
	// 	ImageID:          "ubuntu:latest",
	// 	Command:          "ls -la --color",
	// 	User:             "root",
	// 	WorkingDir:       "/testbed",
	// 	NetworkDisabled:  false,
	// 	TimeoutInSeconds: 5,
	// 	RequestType:      model.REQUEST_TYPE_RUN_COMMAND,
	// }
	// agent.PutRequestToQueue(request)

	// time.Sleep(10 * time.Second)
	// request = model.RolloutRequestInput{
	// 	ID:               "1236",
	// 	TrajectoryID:     "test-trajectory",
	// 	ImageID:          "ubuntu:latest",
	// 	Command:          "git clone https://github.com/nginx/nginx.git && cd nginx",
	// 	User:             "root",
	// 	WorkingDir:       "/testbed",
	// 	NetworkDisabled:  false,
	// 	TimeoutInSeconds: 5,
	// 	RequestType:      model.REQUEST_TYPE_RUN_COMMAND,
	// }
	// agent.PutRequestToQueue(request)

	// time.Sleep(5 * time.Second)
	// request = model.RolloutRequestInput{
	// 	ID:               "1237",
	// 	TrajectoryID:     "test-trajectory",
	// 	ImageID:          "ubuntu:latest",
	// 	Command:          "rm README.md",
	// 	User:             "root",
	// 	WorkingDir:       "/testbed",
	// 	NetworkDisabled:  false,
	// 	TimeoutInSeconds: 5,
	// 	RequestType:      model.REQUEST_TYPE_RUN_COMMAND,
	// }
	// agent.PutRequestToQueue(request)

	// time.Sleep(5 * time.Second)
	// request = model.RolloutRequestInput{
	// 	ID:               "1238",
	// 	TrajectoryID:     "test-trajectory",
	// 	ImageID:          "ubuntu:latest",
	// 	TimeoutInSeconds: 5,
	// 	RequestType:      model.REQUEST_TYPE_GET_PATCH,
	// }
	// agent.PutRequestToQueue(request)

	// // log.Println("Response received from queue")
	// // fmt.Printf("Received response: %+v\n", response)
	// // write response to a file
	// time.Sleep(5 * time.Second) // Simulate some delay before sending the next request
	// request = model.RolloutRequestInput{
	// 	ID:           "1239",
	// 	TrajectoryID: "test-trajectory",
	// 	RequestType:  model.REQUEST_TYPE_GET_OUTPUT,
	// }
	// agent.PutRequestToQueue(request)

	select {} // Block indefinitely
}

func writeResponseToFile(response model.RolloutResponse) {
	// responseJSON, err := json.MarshalIndent(response, "", "  ")
	// if err != nil {
	// 	log.Fatalf("Error converting response to JSON: %v", err)
	// }

	// // Write the response to a file
	// err = ioutil.WriteFile(fmt.Sprintf("response-%s.json", response.ID), responseJSON, 0644)
	// if err != nil {
	// 	log.Fatalf("Error writing response to file: %v", err)
	// }

	// log.Printf("Response saved to response.json: ID=%s, TrajectoryID=%s, ExitCode=%d",
	// 	response.ID, response.TrajectoryID, response.ExitCode)
}

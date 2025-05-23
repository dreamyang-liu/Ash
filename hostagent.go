package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"github.com/multiturn-rl-hostagent/docker"
	"github.com/multiturn-rl-hostagent/model"
	"github.com/multiturn-rl-hostagent/monitor"
	"github.com/multiturn-rl-hostagent/queue"
)

// HostAgent represents the main agent that manages Docker containers and monitors resources
type HostAgent struct {
	dockerManager *docker.Manager
	hostMonitor   *monitor.HostMonitor
	queueClient   *queue.RabbitMQClient
	requestQueue  chan model.RolloutRequest
	responseQueue chan model.RolloutResponse
	ctx           context.Context
	cancel        context.CancelFunc
}

// NewHostAgent creates a new instance of HostAgent
func NewHostAgent() (*HostAgent, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Initialize request and response queues
	requestQueue := make(chan model.RolloutRequest, 50)
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

// Start begins the host agent operations
func (ha *HostAgent) Start() error {
	log.Println("Starting Host Agent...")

	go ha.dockerManager.Start()

	// Wait for termination signal
	// sigCh := make(chan os.Signal, 1)
	// signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	// <-sigCh

	// log.Println("Shutting down Host Agent...")
	// ha.Shutdown()
	return nil
}

// Shutdown stops all agent operations
func (ha *HostAgent) Shutdown() {
	ha.cancel()

	// Close RabbitMQ connection
	if ha.queueClient != nil {
		ha.queueClient.Close()
	}

	// Cleanup any running containers
	if ha.dockerManager != nil {
		ha.dockerManager.CleanupAllContainers(ha.ctx)
	}

	log.Println("Host Agent shutdown complete")
}

func (ha *HostAgent) PutRequestToQueue(request model.RolloutRequest) {
	ha.requestQueue <- request
}

func (ha *HostAgent) GetResponseFromQueue() model.RolloutResponse {
	response := <-ha.responseQueue
	return response
}

func main() {
	log.Println("Initializing host agent...")
	agent, err := NewHostAgent()
	if err != nil {
		log.Fatalf("Failed to initialize host agent: %v", err)
	}
	log.Println("Host agent initialized successfully")

	// Start the agent in a goroutine so we can continue execution
	if err := agent.Start(); err != nil {
		log.Fatalf("Host agent error: %v", err)
	}

	log.Println("Preparing rollout request...")
	// Example usage of request and response queues
	const CONTAINER_INIT_COMMAND = "git fetch && " +
		"git checkout %s && " +
		"git branch -D main master || true && " +
		"git remote remove origin || true && " +
		"git checkout -b main"
	request := model.RolloutRequest{
		ID:               "123",
		TrajectoryID:     "test-trajectory",
		ImageID:          "ubuntu:latest",
		Command:          "apt-get -y update && apt-get install -y git",
		User:             "root",
		WorkingDir:       "/testbed",
		NetworkDisabled:  false,
		TimeoutInSeconds: 30,
	}

	log.Println("Sending request to queue...")
	agent.PutRequestToQueue(request)
	log.Println("Request sent to queue, waiting for response...")

	go func() {
		for {
			response := agent.GetResponseFromQueue()
			writeResponseToFile(response)
		}
	}()
	// log.Println("Response received from queue")
	// fmt.Printf("Received response: %+v\n", response)
	// write response to a file
	time.Sleep(30 * time.Second) // Simulate some delay before sending the next request
	request2 := model.RolloutRequest{
		ID:               "1234",
		TrajectoryID:     "test-trajectory",
		ImageID:          "ubuntu:latest",
		Command:          "ls -la",
		User:             "root",
		WorkingDir:       "/testbed",
		NetworkDisabled:  false,
		TimeoutInSeconds: 5,
	}
	agent.PutRequestToQueue(request2)

	select {} // Block indefinitely
}

func writeResponseToFile(response model.RolloutResponse) {
	responseJSON, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		log.Fatalf("Error converting response to JSON: %v", err)
	}

	// Write the response to a file
	err = ioutil.WriteFile(fmt.Sprintf("response-%s.json", response.ID), responseJSON, 0644)
	if err != nil {
		log.Fatalf("Error writing response to file: %v", err)
	}

	log.Printf("Response saved to response.json: ID=%s, TrajectoryID=%s, ExitCode=%d",
		response.ID, response.TrajectoryID, response.ExitCode)
}

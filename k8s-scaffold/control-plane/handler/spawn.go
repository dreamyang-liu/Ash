package handler

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"k8s.io/client-go/kubernetes"

	"github.com/rl-sandbox/k8s-cp/config"
	"github.com/rl-sandbox/k8s-cp/k8s"
	"github.com/rl-sandbox/k8s-cp/metrics"
	"github.com/rl-sandbox/k8s-cp/store"
)

// Port represents a container port
type Port struct {
	ContainerPort int `json:"container_port"`
}

// SpawnReq represents a sandbox creation request
type SpawnReq struct {
	Image        string            `json:"image" binding:"required"`
	Name         string            `json:"name"`
	Ports        []Port            `json:"ports"`
	Env          map[string]string `json:"env"`
	Resources    ResourceReq       `json:"resources"`
	NodeSelector map[string]string `json:"node_selector"`
	Path         string            `json:"path"` // endpoint path: "/mcp" (default) or "/jsonrpc"
}

// ResourceReq specifies resource requirements
type ResourceReq struct {
	Requests ResourceSpec `json:"requests"`
	Limits   ResourceSpec `json:"limits"`
}

// ResourceSpec specifies CPU and memory
type ResourceSpec struct {
	CPU    string `json:"cpu"`
	Memory string `json:"memory"`
}

// SpawnResp represents a sandbox creation response
type SpawnResp struct {
	Name             string `json:"name"`
	UUID             string `json:"uuid"`
	Namespace        string `json:"namespace"`
	Status           string `json:"status"`
	ServiceType      string `json:"service_type"`
	ClusterIP        string `json:"cluster_ip,omitempty"`
	Host             string `json:"host,omitempty"`
	ExternalIP       string `json:"external_ip,omitempty"`
	ExternalHostname string `json:"external_hostname,omitempty"`
	Ports            []int  `json:"ports,omitempty"`
	NodePorts        []int  `json:"node_ports,omitempty"`
	Message          string `json:"message,omitempty"`
}

// Dependencies holds handler dependencies
type Dependencies struct {
	Clientset *kubernetes.Clientset
	Redis     *redis.Client
	Config    *config.Config
}

// HandleSpawn handles POST /create requests
func HandleSpawn(deps Dependencies) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		var req SpawnReq
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
		defer cancel()

		// Generate name if not provided
		name := req.Name
		if name == "" {
			name = fmt.Sprintf("sandbox-%s", randSuffix(12))
		}

		// Convert ports
		var ports []int
		for _, p := range req.Ports {
			ports = append(ports, p.ContainerPort)
		}

		// Create sandbox
		sandboxReq := k8s.SandboxRequest{
			Image:        req.Image,
			Name:         name,
			Ports:        ports,
			Env:          req.Env,
			NodeSelector: req.NodeSelector,
			Resources: k8s.ResourceRequirements{
				Requests: k8s.ResourceSpec{
					CPU:    req.Resources.Requests.CPU,
					Memory: req.Resources.Requests.Memory,
				},
				Limits: k8s.ResourceSpec{
					CPU:    req.Resources.Limits.CPU,
					Memory: req.Resources.Limits.Memory,
				},
			},
		}

		result, err := k8s.CreateSandbox(ctx, deps.Clientset, deps.Config, sandboxReq)
		if err != nil {
			log.Printf("Failed to create sandbox: %v", err)
			metrics.SandboxCreateTotal.WithLabelValues("error").Inc()
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to create sandbox: %v", err)})
			return
		}

		// Generate UUID and save to Redis
		sandboxUUID := fmt.Sprintf("%s-%s", name, uuid.New().String())
		sandboxStatus := "ready"
		if !result.Ready {
			sandboxStatus = "starting"
		}
		if result.Existed {
			sandboxStatus = "exists"
		}

		sandboxPort := 0
		if len(result.Ports) > 0 {
			sandboxPort = result.Ports[0]
		}

		path := req.Path
		if path == "" {
			path = "/mcp"
		}

		record := store.SandboxRecord{
			UUID:      sandboxUUID,
			Host:      fmt.Sprintf("%s.%s.svc.cluster.local", name, deps.Config.Namespace),
			Port:      sandboxPort,
			Status:    sandboxStatus,
			Name:      name,
			Namespace: deps.Config.Namespace,
			Path:      path,
		}

		if err := store.SaveRecord(ctx, deps.Redis, record); err != nil {
			log.Printf("Failed to save sandbox record to Redis: %v", err)
		}

		log.Printf("Sandbox created: name=%s, uuid=%s, status=%s", name, sandboxUUID, sandboxStatus)

		// Record metrics
		metrics.SandboxCreateTotal.WithLabelValues("success").Inc()
		metrics.SandboxCreateDuration.Observe(time.Since(start).Seconds())

		resp := SpawnResp{
			Name:        name,
			UUID:        sandboxUUID,
			Namespace:   deps.Config.Namespace,
			Status:      cases.Title(language.English).String(sandboxStatus),
			ServiceType: "ClusterIP",
			ClusterIP:   result.ClusterIP,
			Host:        fmt.Sprintf("%s.%s.svc.cluster.local", name, deps.Config.Namespace),
			Ports:       result.Ports,
		}

		c.JSON(http.StatusOK, resp)
	}
}

// randSuffix generates a random string suffix
func randSuffix(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

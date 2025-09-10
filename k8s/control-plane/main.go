package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

func init() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())
}

type Port struct {
	ContainerPort int `json:"container_port"`
}

type SpawnReq struct {
	Image    string            `json:"image" binding:"required"`
	Name     string            `json:"name"`
	Ports    []Port            `json:"ports"`
	Expose   string            `json:"expose" binding:"required"`
	Replicas int               `json:"replicas" binding:"required"`
	Env      map[string]string `json:"env"`
}

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

// Configuration holds all the environment-based configuration
type Config struct {
	Namespace          string
	WaitDeployReadySec int
	WaitSvcIPSec       int
	RedisHost          string
	RedisPort          int
	RedisDB            int
	SandboxMaxTTLSec   int64
}

// LoadConfig loads configuration from environment variables
func LoadConfig() *Config {
	namespace := os.Getenv("TARGET_NAMESPACE")
	if namespace == "" {
		namespace = "apps"
	}

	waitDeployReadySec := 120
	if v := os.Getenv("WAIT_DEPLOY_READY_SEC"); v != "" {
		fmt.Sscanf(v, "%d", &waitDeployReadySec)
	}

	waitSvcIPSec := 120
	if v := os.Getenv("WAIT_SVC_IP_SEC"); v != "" {
		fmt.Sscanf(v, "%d", &waitSvcIPSec)
	}

	redisHost := os.Getenv("REDIS_HOST")
	if redisHost == "" {
		redisHost = "localhost"
	}

	redisPort := 6379
	if v := os.Getenv("REDIS_PORT"); v != "" {
		fmt.Sscanf(v, "%d", &redisPort)
	}

	redisDB := 0
	if v := os.Getenv("REDIS_DB"); v != "" {
		fmt.Sscanf(v, "%d", &redisDB)
	}

	sandboxMaxTTLSec := int64(3600)
	if v := os.Getenv("SANDBOX_MAX_TTL_SEC"); v != "" {
		fmt.Sscanf(v, "%d", &sandboxMaxTTLSec)
	}

	return &Config{
		Namespace:          namespace,
		WaitDeployReadySec: waitDeployReadySec,
		WaitSvcIPSec:       waitSvcIPSec,
		RedisHost:          redisHost,
		RedisPort:          redisPort,
		RedisDB:            redisDB,
		SandboxMaxTTLSec:   sandboxMaxTTLSec,
	}
}

// Generate a random string of specified length
func randSuffix(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

// Get Kubernetes client from in-cluster or kubeconfig
func getK8sClient() (*kubernetes.Clientset, error) {
	var config *rest.Config
	var err error

	// Try in-cluster config first
	config, err = rest.InClusterConfig()
	if err != nil {
		// Fall back to kubeconfig
		kubeconfig := os.Getenv("KUBECONFIG")
		if kubeconfig == "" {
			kubeconfig = os.ExpandEnv("$HOME/.kube/config")
		}
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create k8s config: %w", err)
		}
	}

	// Create clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create k8s client: %w", err)
	}

	return clientset, nil
}

// Create a Redis client
func createRedisClient(config *Config) *redis.Client {
	return redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", config.RedisHost, config.RedisPort),
		DB:   config.RedisDB,
	})
}

func main() {
	// Load configuration
	config := LoadConfig()

	// Create Redis client
	rdb := createRedisClient(config)
	defer rdb.Close()

	// Ping Redis to ensure connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}

	// Set up Gin router
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(gin.Logger())

	// Health check endpoints
	r.GET("/healthz", func(c *gin.Context) {
		c.String(http.StatusOK, "ok")
	})

	r.GET("/readyz", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 500*time.Millisecond)
		defer cancel()

		if err := rdb.Ping(ctx).Err(); err != nil {
			c.String(http.StatusServiceUnavailable, "redis not ready")
			return
		}

		c.String(http.StatusOK, "ready")
	})

	// Main API endpoints
	r.POST("/spawn", func(c *gin.Context) {
		var req SpawnReq
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Use request context with timeout
		ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
		defer cancel()

		name := req.Name
		if name == "" {
			name = fmt.Sprintf("sandbox-%s", randSuffix(12))
		}
		labels := map[string]string{"app": name, "from": "control-plane", "type": "sandbox"}

		// 1) Deployment
		var envVars []corev1.EnvVar
		for k, v := range req.Env {
			envVars = append(envVars, corev1.EnvVar{Name: k, Value: v})
		}

		var containerPorts []corev1.ContainerPort
		for _, p := range req.Ports {
			containerPorts = append(containerPorts, corev1.ContainerPort{ContainerPort: int32(p.ContainerPort)})
		}
		if len(containerPorts) == 0 {
			containerPorts = append(containerPorts, corev1.ContainerPort{ContainerPort: 80})
		}

		// Create container with probes
		container := corev1.Container{
			Name:  "sandbox",
			Image: req.Image,
			Ports: containerPorts,
			Env:   envVars,
		}
		podSpec := corev1.PodSpec{
			Containers:         []corev1.Container{container},
			ServiceAccountName: "control-plane",
			NodeSelector: map[string]string{
				"eks.amazonaws.com/nodegroup": "sandbox",
			},
		}
		dep := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: config.Namespace,
				Labels:    labels,
			},
			Spec: appsv1.DeploymentSpec{
				Replicas: int32Ptr(req.Replicas),
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"app": name},
				},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: labels},
					Spec:       podSpec,
				},
			},
		}

		clientset, err := getK8sClient()
		if err != nil {
			log.Printf("Failed to get k8s client: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to connect to Kubernetes"})
			return
		}

		// Create deployment with context
		_, err = clientset.AppsV1().Deployments(config.Namespace).Create(ctx, dep, metav1.CreateOptions{})
		if err != nil {
			log.Printf("Failed to create deployment: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to create deployment: %v", err)})
			return
		}

		// 2) Service (optional)
		var svcObj *corev1.Service
		if req.Expose == "LoadBalancer" || req.Expose == "NodePort" || req.Expose == "ClusterIP" {
			var svcPorts []corev1.ServicePort
			for _, p := range req.Ports {
				svcPorts = append(svcPorts, corev1.ServicePort{
					Port:       int32(p.ContainerPort),
					TargetPort: intstrFromInt(p.ContainerPort),
				})
			}
			if len(svcPorts) == 0 {
				svcPorts = append(svcPorts, corev1.ServicePort{
					Port:       80,
					TargetPort: intstrFromInt(80),
				})
			}
			svc := &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: config.Namespace,
					Labels:    labels,
				},
				Spec: corev1.ServiceSpec{
					Type:     corev1.ServiceType(req.Expose),
					Selector: map[string]string{"app": name},
					Ports:    svcPorts,
				},
			}
			svcObj, err = clientset.CoreV1().Services(config.Namespace).Create(context.Background(), svc, metav1.CreateOptions{})
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
		}

		// 3) Wait for Deployment Ready with exponential backoff
		ready := false
		backoff := 1 * time.Second
		maxBackoff := 10 * time.Second
		end := time.Now().Add(time.Duration(config.WaitDeployReadySec) * time.Second)

		for time.Now().Before(end) {
			cur, err := clientset.AppsV1().Deployments(config.Namespace).Get(ctx, name, metav1.GetOptions{})
			if err == nil && cur.Status.AvailableReplicas >= 1 {
				ready = true
				break
			}

			// Use exponential backoff with jitter
			jitter := time.Duration(rand.Int63n(int64(backoff) / 2))
			sleepTime := backoff + jitter
			time.Sleep(sleepTime)

			// Increase backoff for next iteration
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}

		// 4) Collect Service Address
		var clusterIP, externalIP, externalHost string
		var svcPorts []int
		var nodePorts []int
		if svcObj != nil {
			end := time.Now().Add(time.Duration(config.WaitSvcIPSec) * time.Second)
			for time.Now().Before(end) {
				s, err := clientset.CoreV1().Services(config.Namespace).Get(context.Background(), name, metav1.GetOptions{})
				if err == nil {

					if len(svcPorts) == 0 {
						for _, p := range s.Spec.Ports {
							svcPorts = append(svcPorts, int(p.Port))
						}
					}
					clusterIP = s.Spec.ClusterIP
					switch req.Expose {
					case "LoadBalancer":
						if s.Status.LoadBalancer.Ingress != nil && len(s.Status.LoadBalancer.Ingress) > 0 {
							ing := s.Status.LoadBalancer.Ingress[0]
							externalIP = ing.IP
							externalHost = ing.Hostname
							break
						}
					case "NodePort":
						if len(nodePorts) == 0 {
							for _, p := range s.Spec.Ports {
								nodePorts = append(nodePorts, int(p.NodePort))
							}
						}
						break
					case "ClusterIP":
						break
					}
					if req.Expose == "ClusterIP" {
						break
					}
				}
				time.Sleep(1 * time.Second)
			}
		}

		// Prepare Redis record
		sandboxUUID := fmt.Sprintf("%s-%s", name, uuid.New().String())

		sandboxStatus := "ready"
		if !ready {
			sandboxStatus = "starting"
		}

		sandboxIP := externalIP
		if sandboxIP == "" {
			sandboxIP = clusterIP
		}

		sandboxPort := 0
		if len(nodePorts) > 0 {
			sandboxPort = nodePorts[0]
		} else if len(svcPorts) > 0 {
			sandboxPort = svcPorts[0]
		}

		expireTime := time.Now().Unix() + config.SandboxMaxTTLSec

		// Create Redis record with pipeline for efficiency
		record := map[string]interface{}{
			"uuid":        sandboxUUID,
			"host":        fmt.Sprintf("%s.%s.svc.cluster.local", name, config.Namespace),
			"port":        sandboxPort,
			"status":      sandboxStatus,
			"expire_time": expireTime,
		}

		key := fmt.Sprintf("sandbox:%s", sandboxUUID)
		pipe := rdb.Pipeline()
		pipe.HSet(ctx, key, record)
		pipe.Expire(ctx, key, time.Duration(config.SandboxMaxTTLSec)*time.Second)

		if _, err := pipe.Exec(ctx); err != nil {
			log.Printf("Failed to save sandbox record to Redis: %v", err)
		}

		log.Printf("Sandbox created: name=%s, uuid=%s, status=%s", name, sandboxUUID, sandboxStatus)

		message := ""
		if externalIP == "" && externalHost == "" && len(nodePorts) == 0 && req.Expose != "None" {
			message = "External address pending"
		}

		resp := SpawnResp{
			Name:             name,
			UUID:             sandboxUUID,
			Namespace:        config.Namespace,
			Status:           strings.Title(sandboxStatus),
			ServiceType:      req.Expose,
			ClusterIP:        clusterIP,
			Host:             fmt.Sprintf("%s.%s.svc.cluster.local", name, config.Namespace),
			ExternalIP:       externalIP,
			ExternalHostname: externalHost,
			Ports:            svcPorts,
			NodePorts:        nodePorts,
			Message:          message,
		}

		// Log status
		status := "success"
		if !ready {
			status = "partial"
		}
		log.Printf("Spawn request completed with status: %s", status)

		c.JSON(http.StatusOK, resp)
	})

	r.DELETE("/deprovision-all", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Minute)
		defer cancel()

		clientset, err := getK8sClient()
		if err != nil {
			log.Printf("Failed to get k8s client: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to connect to Kubernetes"})
			return
		}

		var succeeded []string
		var failed []string

		// Find all deployments created by control-plane with label type=sandbox
		selector := "from=control-plane,type=sandbox"
		deps, err := clientset.AppsV1().Deployments(config.Namespace).List(ctx, metav1.ListOptions{
			LabelSelector: selector,
		})
		if err != nil {
			log.Printf("Failed to list deployments: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list deployments"})
			return
		}

		for _, dep := range deps.Items {
			name := dep.Name
			namespace := dep.Namespace
			id := fmt.Sprintf("%s/%s", namespace, name)

			// Delete service
			if err := clientset.CoreV1().Services(namespace).Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
				// Log but continue
				log.Printf("Failed to delete service %s: %v", id, err)
			}

			// Delete deployment
			if err := clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
				log.Printf("Failed to delete deployment %s: %v", id, err)
			}

			// Remove associated Redis keys: sandbox:<name>-*
			pattern := fmt.Sprintf("sandbox:%s-*", name)
			iter := rdb.Scan(ctx, 0, pattern, 0).Iterator()
			var redisDelErr bool
			var anyDeleted bool
			for iter.Next(ctx) {
				key := iter.Val()
				anyDeleted = true
				if err := rdb.Del(ctx, key).Err(); err != nil {
					log.Printf("Failed to delete Redis key %s for %s: %v", key, id, err)
					redisDelErr = true
				}
			}
			if err := iter.Err(); err != nil {
				log.Printf("Error scanning Redis for pattern %s: %v", pattern, err)
				redisDelErr = true
			}
			// If no matching redis key found, that's not a fatal error; still consider succeeded.
			if redisDelErr {
				failed = append(failed, id)
			} else {
				// Consider this resource successfully handled
				succeeded = append(succeeded, id)
				// If there were no redis keys but resource deletions succeeded, still success.
				if !anyDeleted {
					log.Printf("No Redis keys found for %s (pattern %s)", id, pattern)
				}
			}
		}

		log.Printf("Deprovision-all completed: succeeded=%d failed=%d", len(succeeded), len(failed))
		c.JSON(http.StatusOK, gin.H{
			"deleted": succeeded,
			"failed":  failed,
			"count":   len(succeeded),
		})
	})

	r.DELETE("/deprovision/:uuid", func(c *gin.Context) {
		uuid := c.Param("uuid")

		// Use request context with timeout
		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
		defer cancel()

		key := fmt.Sprintf("sandbox:%s", uuid)
		result, err := rdb.HGetAll(ctx, key).Result()
		if err != nil || len(result) == 0 {
			log.Printf("Deprovision failed: UUID %s not found", uuid)
			c.JSON(http.StatusNotFound, gin.H{"error": "UUID not found"})
			return
		}

		name := result["host"]

		parts := strings.Split(name, ".")
		if len(parts) < 2 {
			log.Printf("Deprovision failed: Invalid host format for UUID %s", uuid)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid host format"})
			return
		}
		svcName := parts[0]
		namespace := parts[1]

		clientset, err := getK8sClient()
		if err != nil {
			log.Printf("Failed to get k8s client: %v", err)
			log.Printf("Deprovision failed: Kubernetes client error")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to connect to Kubernetes"})
			return
		}

		// Delete resources sequentially
		if err := clientset.CoreV1().Services(namespace).Delete(ctx, svcName, metav1.DeleteOptions{}); err != nil {
			log.Printf("Failed to delete service %s: %v", svcName, err)
		}

		if err := clientset.AppsV1().Deployments(namespace).Delete(ctx, svcName, metav1.DeleteOptions{}); err != nil {
			log.Printf("Failed to delete deployment %s: %v", svcName, err)
		}

		// Delete Redis key
		if err := rdb.Del(ctx, key).Err(); err != nil {
			log.Printf("Failed to delete Redis key %s: %v", key, err)
		}

		log.Printf("Successfully deprovisioned UUID %s", uuid)
		c.JSON(http.StatusOK, gin.H{"message": "Deprovisioned", "uuid": uuid})
	})
	// Create HTTP server with graceful shutdown
	srv := http.Server{
		Addr:    ":8080",
		Handler: r,
	}

	// Start server in a goroutine
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Create shutdown context with timeout
	ctx, cancel = context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Shutdown the server
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited properly")
}

func int32Ptr(i int) *int32 {
	v := int32(i)
	return &v
}

func intstrFromInt(i int) intstr.IntOrString {
	return intstr.IntOrString{Type: intstr.Int, IntVal: int32(i)}
}

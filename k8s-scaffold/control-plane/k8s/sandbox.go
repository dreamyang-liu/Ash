package k8s

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"

	"github.com/rl-sandbox/k8s-cp/config"
)

// SandboxRequest contains the parameters for creating a sandbox
type SandboxRequest struct {
	Image        string
	Name         string
	Ports        []int
	Env          map[string]string
	Resources    ResourceRequirements
	NodeSelector map[string]string
}

// ResourceRequirements specifies CPU and memory requests/limits
type ResourceRequirements struct {
	Requests ResourceSpec
	Limits   ResourceSpec
}

// ResourceSpec specifies CPU and memory
type ResourceSpec struct {
	CPU    string
	Memory string
}

// SandboxResult contains the result of creating a sandbox
type SandboxResult struct {
	Name      string
	Namespace string
	ClusterIP string
	Ports     []int
	Ready     bool
	Existed   bool
}

// CreateSandbox creates a Kubernetes Deployment and Service for a sandbox
func CreateSandbox(ctx context.Context, clientset *kubernetes.Clientset, cfg *config.Config, req SandboxRequest) (*SandboxResult, error) {
	labels := map[string]string{
		"app":  req.Name,
		"from": "control-plane",
		"type": "sandbox",
	}

	// Build environment variables
	var envVars []corev1.EnvVar
	for k, v := range req.Env {
		envVars = append(envVars, corev1.EnvVar{Name: k, Value: v})
	}

	// Build container ports
	var containerPorts []corev1.ContainerPort
	for _, p := range req.Ports {
		containerPorts = append(containerPorts, corev1.ContainerPort{ContainerPort: int32(p)})
	}
	if len(containerPorts) == 0 {
		containerPorts = append(containerPorts, corev1.ContainerPort{ContainerPort: 80})
	}

	// Determine probe port
	probePort := 3000
	if len(containerPorts) > 0 {
		probePort = int(containerPorts[0].ContainerPort)
	}

	// Create container with readiness probe
	container := corev1.Container{
		Name:  "sandbox",
		Image: req.Image,
		Ports: containerPorts,
		Env:   envVars,
		ReadinessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(probePort),
				},
			},
			InitialDelaySeconds: 2,
			PeriodSeconds:       3,
			TimeoutSeconds:      1,
			SuccessThreshold:    1,
			FailureThreshold:    10,
		},
	}

	// Add resource limits and requests
	if req.Resources.Requests.CPU != "" || req.Resources.Requests.Memory != "" ||
		req.Resources.Limits.CPU != "" || req.Resources.Limits.Memory != "" {

		container.Resources = corev1.ResourceRequirements{}

		if req.Resources.Requests.CPU != "" || req.Resources.Requests.Memory != "" {
			container.Resources.Requests = corev1.ResourceList{}
			if req.Resources.Requests.CPU != "" {
				qty, err := resource.ParseQuantity(req.Resources.Requests.CPU)
				if err != nil {
					return nil, fmt.Errorf("invalid CPU request: %w", err)
				}
				container.Resources.Requests[corev1.ResourceCPU] = qty
			}
			if req.Resources.Requests.Memory != "" {
				qty, err := resource.ParseQuantity(req.Resources.Requests.Memory)
				if err != nil {
					return nil, fmt.Errorf("invalid memory request: %w", err)
				}
				container.Resources.Requests[corev1.ResourceMemory] = qty
			}
		}

		if req.Resources.Limits.CPU != "" || req.Resources.Limits.Memory != "" {
			container.Resources.Limits = corev1.ResourceList{}
			if req.Resources.Limits.CPU != "" {
				qty, err := resource.ParseQuantity(req.Resources.Limits.CPU)
				if err != nil {
					return nil, fmt.Errorf("invalid CPU limit: %w", err)
				}
				container.Resources.Limits[corev1.ResourceCPU] = qty
			}
			if req.Resources.Limits.Memory != "" {
				qty, err := resource.ParseQuantity(req.Resources.Limits.Memory)
				if err != nil {
					return nil, fmt.Errorf("invalid memory limit: %w", err)
				}
				container.Resources.Limits[corev1.ResourceMemory] = qty
			}
		}
	}

	// Node selector
	nodeSelector := req.NodeSelector
	if nodeSelector == nil {
		nodeSelector = map[string]string{
			"kubernetes.io/os": "linux",
		}
	}

	// Create deployment
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      req.Name,
			Namespace: cfg.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: int32Ptr(1),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": req.Name},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers:         []corev1.Container{container},
					ServiceAccountName: cfg.ServiceAccountName,
					NodeSelector:       nodeSelector,
				},
			},
		},
	}

	existed := false
	_, err := clientset.AppsV1().Deployments(cfg.Namespace).Create(ctx, dep, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			// Handle idempotent create: fetch existing deployment
			log.Printf("Deployment %s already exists, treating as idempotent create", req.Name)
			existed = true
		} else {
			return nil, fmt.Errorf("failed to create deployment: %w", err)
		}
	}

	// Create service
	var servicePorts []corev1.ServicePort
	for _, p := range req.Ports {
		servicePorts = append(servicePorts, corev1.ServicePort{
			Port:       int32(p),
			TargetPort: intstr.FromInt(p),
		})
	}
	if len(servicePorts) == 0 {
		servicePorts = append(servicePorts, corev1.ServicePort{
			Port:       80,
			TargetPort: intstr.FromInt(80),
		})
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      req.Name,
			Namespace: cfg.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:     corev1.ServiceTypeClusterIP,
			Selector: map[string]string{"app": req.Name},
			Ports:    servicePorts,
		},
	}

	svcObj, err := clientset.CoreV1().Services(cfg.Namespace).Create(ctx, svc, metav1.CreateOptions{})
	if err != nil {
		if errors.IsAlreadyExists(err) {
			// Service already exists, fetch it
			svcObj, err = clientset.CoreV1().Services(cfg.Namespace).Get(ctx, req.Name, metav1.GetOptions{})
			if err != nil {
				return nil, fmt.Errorf("failed to get existing service: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to create service: %w", err)
		}
	}

	// Wait for deployment ready
	ready := WaitReady(ctx, clientset, cfg.Namespace, req.Name, cfg.WaitDeployReadySec)

	// Collect service info
	var clusterIP string
	var svcPorts []int
	if svcObj != nil {
		s, err := clientset.CoreV1().Services(cfg.Namespace).Get(ctx, req.Name, metav1.GetOptions{})
		if err == nil {
			clusterIP = s.Spec.ClusterIP
			for _, p := range s.Spec.Ports {
				svcPorts = append(svcPorts, int(p.Port))
			}
		}
	}

	return &SandboxResult{
		Name:      req.Name,
		Namespace: cfg.Namespace,
		ClusterIP: clusterIP,
		Ports:     svcPorts,
		Ready:     ready,
		Existed:   existed,
	}, nil
}

// DestroySandbox deletes a Kubernetes Deployment and Service
func DestroySandbox(ctx context.Context, clientset *kubernetes.Clientset, namespace, name string) error {
	// Delete service
	if err := clientset.CoreV1().Services(namespace).Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
		if !errors.IsNotFound(err) {
			log.Printf("Failed to delete service %s/%s: %v", namespace, name, err)
		}
	}

	// Delete deployment
	if err := clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
		if !errors.IsNotFound(err) {
			log.Printf("Failed to delete deployment %s/%s: %v", namespace, name, err)
		}
	}

	return nil
}

// WaitReady waits for a deployment to become ready with exponential backoff
func WaitReady(ctx context.Context, clientset *kubernetes.Clientset, namespace, name string, timeoutSec int) bool {
	backoff := 1 * time.Second
	maxBackoff := 10 * time.Second
	end := time.Now().Add(time.Duration(timeoutSec) * time.Second)

	for time.Now().Before(end) {
		cur, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil && cur.Status.AvailableReplicas >= 1 {
			return true
		}

		// Exponential backoff with jitter
		jitter := time.Duration(rand.Int63n(int64(backoff) / 2))
		sleepTime := backoff + jitter
		time.Sleep(sleepTime)

		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}

	return false
}

func int32Ptr(i int) *int32 {
	v := int32(i)
	return &v
}

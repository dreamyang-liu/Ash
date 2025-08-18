package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
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

func randSuffix(n int) string {
	letters := []rune("abcdefghijklmnopqrstuvwxyz0123456789")
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

func getK8sClient() (*kubernetes.Clientset, error) {
	var config *rest.Config
	var err error
	config, err = rest.InClusterConfig()
	if err != nil {
		kubeconfig := os.Getenv("KUBECONFIG")
		if kubeconfig == "" {
			kubeconfig = os.ExpandEnv("$HOME/.kube/config")
		}
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			return nil, err
		}
	}
	return kubernetes.NewForConfig(config)
}

func main() {
	r := gin.Default()
	r.POST("/spawn", func(c *gin.Context) {
		var req SpawnReq
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		namespace := os.Getenv("TARGET_NAMESPACE")
		if namespace == "" {
			namespace = "apps"
		}
		waitDeployReadySec := 120
		waitSvcIPSec := 120
		if v := os.Getenv("WAIT_DEPLOY_READY_SEC"); v != "" {
			fmt.Sscanf(v, "%d", &waitDeployReadySec)
		}
		if v := os.Getenv("WAIT_SVC_IP_SEC"); v != "" {
			fmt.Sscanf(v, "%d", &waitSvcIPSec)
		}

		name := req.Name
		if name == "" {
			name = fmt.Sprintf("sandbox-%s", randSuffix(12))
		}
		labels := map[string]string{"app": name, "from": "spawner"}

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
		container := corev1.Container{
			Name:  "sandbox",
			Image: req.Image,
			Ports: containerPorts,
			Env:   envVars,
		}
		podSpec := corev1.PodSpec{
			Containers:         []corev1.Container{container},
			ServiceAccountName: "spawner",
		}
		dep := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: namespace,
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
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		_, err = clientset.AppsV1().Deployments(namespace).Create(context.Background(), dep, metav1.CreateOptions{})
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
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
					Namespace: namespace,
					Labels:    labels,
				},
				Spec: corev1.ServiceSpec{
					Type:     corev1.ServiceType(req.Expose),
					Selector: map[string]string{"app": name},
					Ports:    svcPorts,
				},
			}
			svcObj, err = clientset.CoreV1().Services(namespace).Create(context.Background(), svc, metav1.CreateOptions{})
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
		}

		// 3) Wait for Deployment Ready
		ready := false
		end := time.Now().Add(time.Duration(waitDeployReadySec) * time.Second)
		for time.Now().Before(end) {
			cur, err := clientset.AppsV1().Deployments(namespace).Get(context.Background(), name, metav1.GetOptions{})
			if err == nil && cur.Status.AvailableReplicas >= 1 {
				ready = true
				break
			}
			time.Sleep(1 * time.Second)
		}

		// 4) Collect Service Address
		var clusterIP, externalIP, externalHost string
		var svcPorts []int
		var nodePorts []int
		if svcObj != nil {
			end := time.Now().Add(time.Duration(waitSvcIPSec) * time.Second)
			for time.Now().Before(end) {
				s, err := clientset.CoreV1().Services(namespace).Get(context.Background(), name, metav1.GetOptions{})
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

		// Redis
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
		rds := redis.NewClient(&redis.Options{
			Addr: fmt.Sprintf("%s:%d", redisHost, redisPort),
			DB:   redisDB,
		})
		ctx := context.Background()

		sandboxStatus := "ready"
		if !ready {
			sandboxStatus = "starting"
		}
		sandboxUUID := fmt.Sprintf("%s-%s", name, uuid.New().String())
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
		expireTime := time.Now().Unix() + 3600
		if v := os.Getenv("SANDBOX_MAX_TTL_SEC"); v != "" {
			var ttl int64
			fmt.Sscanf(v, "%d", &ttl)
			expireTime = time.Now().Unix() + ttl
		}

		record := map[string]interface{}{
			"uuid":        sandboxUUID,
			"host":        fmt.Sprintf("%s.%s.svc.cluster.local", name, namespace),
			"port":        sandboxPort,
			"status":      sandboxStatus,
			"expire_time": expireTime,
		}
		rds.HSet(ctx, fmt.Sprintf("sandbox:%s", sandboxUUID), record)

		log.Printf("Sandbox created: %v", record)

		message := ""
		if externalIP == "" && externalHost == "" && len(nodePorts) == 0 && req.Expose != "None" {
			message = "External address pending"
		}

		resp := SpawnResp{
			Name:             name,
			UUID:             sandboxUUID,
			Namespace:        namespace,
			Status:           strings.Title(sandboxStatus),
			ServiceType:      req.Expose,
			ClusterIP:        clusterIP,
			Host:             fmt.Sprintf("%s.%s.svc.cluster.local", name, namespace),
			ExternalIP:       externalIP,
			ExternalHostname: externalHost,
			Ports:            svcPorts,
			NodePorts:        nodePorts,
			Message:          message,
		}
		c.JSON(200, resp)
	})

	r.DELETE("/deprovision/:uuid", func(c *gin.Context) {
		uuid := c.Param("uuid")
		ctx := context.Background()
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
		rds := redis.NewClient(&redis.Options{
			Addr: fmt.Sprintf("%s:%d", redisHost, redisPort),
			DB:   redisDB,
		})

		key := fmt.Sprintf("sandbox:%s", uuid)
		result, err := rds.HGetAll(ctx, key).Result()
		if err != nil || len(result) == 0 {
			c.JSON(404, gin.H{"error": "UUID not found"})
			return
		}

		name := result["host"]

		parts := strings.Split(name, ".")
		if len(parts) < 2 {
			c.JSON(400, gin.H{"error": "Invalid host format"})
			return
		}
		svcName := parts[0]
		namespace := parts[1]

		clientset, err := getK8sClient()
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}

		_ = clientset.CoreV1().Services(namespace).Delete(ctx, svcName, metav1.DeleteOptions{})

		_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, svcName, metav1.DeleteOptions{})

		_ = rds.Del(ctx, key).Err()

		c.JSON(200, gin.H{"message": "Deprovisioned", "uuid": uuid})
	})
	r.Run(":80")
}

func int32Ptr(i int) *int32 {
	v := int32(i)
	return &v
}

func intstrFromInt(i int) intstr.IntOrString {
	return intstr.IntOrString{Type: intstr.Int, IntVal: int32(i)}
}
